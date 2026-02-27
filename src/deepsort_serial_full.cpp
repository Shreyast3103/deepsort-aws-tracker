// deepsort_serial_full.cpp//with multiple videos
// Serial DeepSORT-like tracker (ONNX Runtime + OpenCV)

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <filesystem>

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

// ---------- Simple Kalman ----------
struct SimpleKalman {
    Mat x; Mat P; Mat Q; Mat R;
    SimpleKalman(float cx=0.0f, float cy=0.0f) {
        x = Mat::zeros(4,1,CV_32F); x.at<float>(0)=cx; x.at<float>(1)=cy;
        P = Mat::eye(4,4,CV_32F) * 500.0f;
        Q = Mat::eye(4,4,CV_32F) * 1.0f;
        R = Mat::eye(2,2,CV_32F) * 10.0f;
    }
    void predict(float dt=1.0f){
        Mat F = Mat::eye(4,4,CV_32F);
        F.at<float>(0,2)=dt; F.at<float>(1,3)=dt;
        x = F * x;
        P = F * P * F.t() + Q;
    }
    void update(float mx, float my){
        Mat H = Mat::zeros(2,4,CV_32F); H.at<float>(0,0)=1; H.at<float>(1,1)=1;
        Mat z = (Mat_<float>(2,1) << mx, my);
        Mat S = H * P * H.t() + R;
        Mat K = P * H.t() * S.inv();
        x = x + K * (z - H * x);
        Mat I = Mat::eye(4,4,CV_32F);
        P = (I - K * H) * P;
    }
    pair<float,float> state_xy() const { return { x.at<float>(0,0), x.at<float>(1,0) }; }
};

// ---------- Utils ----------
static float iou_rect(const Rect2f &A, const Rect2f &B){
    float xA = max(A.x, B.x), yA = max(A.y, B.y);
    float xB = min(A.x + A.width, B.x + B.width);
    float yB = min(A.y + A.height, B.y + B.height);
    float w = max(0.0f, xB - xA), h = max(0.0f, yB - yA);
    float inter = w * h;
    float ua = A.area() + B.area() - inter + 1e-9f;
    return inter / ua;
}
static float cosine_distance(const vector<float> &a, const vector<float> &b){
    if(a.empty() || b.empty() || a.size() != b.size()) return 1.0f;
    double dot=0, na=0, nb=0;
    for(size_t i=0;i<a.size();++i){ dot += (double)a[i]*b[i]; na += (double)a[i]*a[i]; nb += (double)b[i]*b[i]; }
    if(na < 1e-9 || nb < 1e-9) return 1.0f;
    return 1.0f - float(dot / (sqrt(na)*sqrt(nb) + 1e-9));
}

// ---------- Hungarian assignment ----------
vector<pair<int,int>> hungarian_assign(const vector<vector<float>> &cost, float cost_thresh=0.7f){
    int m = (int)cost.size();
    if(m==0) return {};
    int n = (int)cost[0].size();
    if(n==0) return {};
    int N = max(m,n);
    const double INF = 1e9;
    vector<vector<double>> A(N, vector<double>(N, INF));
    for(int i=0;i<m;i++) for(int j=0;j<n;j++) A[i][j] = cost[i][j];
    vector<double> u(N+1,0), v(N+1,0);
    vector<int> p(N+1,0), way(N+1,0);
    for(int i=1;i<=N;i++){
        p[0]=i; int j0=0;
        vector<double> minv(N+1,1e18);
        vector<char> used(N+1, false);
        do{
            used[j0]=true; int i0=p[j0]; int j1=0; double delta=1e18;
            for(int j=1;j<=N;j++) if(!used[j]){
                double cur = A[i0-1][j-1] - u[i0] - v[j];
                if(cur < minv[j]){ minv[j]=cur; way[j]=j0; }
                if(minv[j] < delta){ delta = minv[j]; j1 = j; }
            }
            for(int j=0;j<=N;j++){
                if(used[j]){ u[p[j]] += delta; v[j] -= delta; }
                else minv[j] -= delta;
            }
            j0 = j1;
        } while(p[j0] != 0);
        do{
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while(j0);
    }
    vector<int> assignment(N+1,0);
    for(int j=1;j<=N;j++) assignment[p[j]] = j;
    vector<pair<int,int>> res;
    for(int i=0;i<m;i++){
        int j = assignment[i+1] - 1;
        if(j>=0 && j<n){
            double c = A[i][j];
            if(c <= cost_thresh + 1e-9) res.emplace_back(i,j);
        }
    }
    return res;
}

// ---------- ONNX Runtime wrapper ----------
struct ONNXModel {
    Ort::Env env;
    Ort::Session session;
    Ort::SessionOptions opts;
    Ort::AllocatorWithDefaultOptions allocator;
    vector<string> input_names;
    vector<string> output_names;

    ONNXModel(const string &path) : env(ORT_LOGGING_LEVEL_WARNING, "onnx"), session(nullptr) {
        opts.SetIntraOpNumThreads(1);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
#ifdef _WIN32
        std::wstring wpath(path.begin(), path.end());
        session = Ort::Session(env, wpath.c_str(), opts);
#else
        session = Ort::Session(env, path.c_str(), opts);
#endif
        size_t n_in = session.GetInputCount();
        for(size_t i=0;i<n_in;i++){
            auto name = session.GetInputNameAllocated(i, allocator);
            input_names.push_back(string(name.get()));
        }
        size_t n_out = session.GetOutputCount();
        for(size_t i=0;i<n_out;i++){
            auto name = session.GetOutputNameAllocated(i, allocator);
            output_names.push_back(string(name.get()));
        }
    }
};

// ---------- YOLO ONNX helper ----------
struct YoloONNX {
    ONNXModel *m; int input_w, input_h; float conf_thresh, nms_thresh;
    string in_name, out_name;
    YoloONNX(ONNXModel *model, int w=416, int h=416, float conf_t=0.5f, float nms_t=0.45f)
      : m(model), input_w(w), input_h(h), conf_thresh(conf_t), nms_thresh(nms_t) {
        if(!m->input_names.empty()) in_name = m->input_names[0];
        if(!m->output_names.empty()) out_name = m->output_names[0];
    }

    static void mat_to_chw(const Mat &img, int W, int H, vector<float> &out){
        Mat resized; resize(img, resized, Size(W,H));
        Mat f; resized.convertTo(f, CV_32F, 1.0f/255.0f);
        out.resize(3 * W * H);
        int idx=0;
        for(int c=0;c<3;c++){
            for(int y=0;y<H;y++){
                for(int x=0;x<W;x++){
                    Vec3f v = f.at<Vec3f>(y,x);
                    out[idx++] = v[c];
                }
            }
        }
    }

    vector<tuple<Rect2f,float,int>> detect(const Mat &frame){
        vector<float> input_tensor;
        mat_to_chw(frame, input_w, input_h, input_tensor);
        array<int64_t,4> dims = {1,3,input_h,input_w};
        Ort::MemoryInfo meminfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        Ort::Value in_tensor = Ort::Value::CreateTensor<float>(meminfo, input_tensor.data(), input_tensor.size(), dims.data(), dims.size());
        const char* in_names[] = { in_name.c_str() };
        const char* out_names[] = { out_name.c_str() };
        vector<Ort::Value> outputs = m->session.Run(Ort::RunOptions{nullptr}, in_names, &in_tensor, 1, out_names, 1);
        vector<tuple<Rect2f,float,int>> dets;
        if(outputs.empty()) return dets;
        Ort::Value &outv = outputs[0];
        auto info = outv.GetTensorTypeAndShapeInfo();
        vector<int64_t> shape = info.GetShape();
        float* ptr = outv.GetTensorMutableData<float>();
        int N=0,K=0;
        if(shape.size()==2){ N=(int)shape[0]; K=(int)shape[1]; }
        else if(shape.size()==3 && shape[0]==1){ N=(int)shape[1]; K=(int)shape[2]; }
        else return dets;

        float sx = float(frame.cols) / float(input_w);
        float sy = float(frame.rows) / float(input_h);

        vector<Rect2f> boxes; vector<float> scores; vector<int> classes;
        for(int i=0;i<N;i++){
            if(K<6) continue;
            float x1 = ptr[i*K + 0], y1 = ptr[i*K + 1], x2 = ptr[i*K + 2], y2 = ptr[i*K + 3];
            float conf = ptr[i*K + 4]; int cls = (int)round(ptr[i*K + 5]);
            if(conf < conf_thresh) continue;  // 🚨 filter low confidence
            if(cls != 0) continue;            // 🚨 keep only person class
            x1 *= sx; x2 *= sx; y1 *= sy; y2 *= sy;
            float w = max(0.0f, x2 - x1), h = max(0.0f, y2 - y1);
            boxes.emplace_back(x1, y1, w, h);
            scores.push_back(conf);
            classes.push_back(cls);
        }

        // simple NMS
        vector<int> idx(boxes.size()); iota(idx.begin(), idx.end(), 0);
        sort(idx.begin(), idx.end(), [&](int a,int b){ return scores[a] > scores[b]; });
        vector<char> removed(boxes.size(),0);
        for(size_t i=0;i<idx.size();++i){
            int ii=idx[i]; if(removed[ii]) continue;
            for(size_t j=i+1;j<idx.size();++j){
                int jj=idx[j]; if(removed[jj]) continue;
                if(iou_rect(boxes[ii], boxes[jj]) > nms_thresh) removed[jj]=1;
            }
        }
        for(size_t i=0;i<idx.size();++i){
            int ii = idx[i]; if(!removed[ii]) dets.emplace_back(boxes[ii], scores[ii], 0);
        }
        return dets;
    }
};

// ---------- ReID ONNX ----------
struct ReidONNX {
    ONNXModel *m; int crop_w, crop_h; string in_name, out_name;
    ReidONNX(ONNXModel *model, int w=128, int h=256): m(model), crop_w(w), crop_h(h) {
        if(!m->input_names.empty()) in_name = m->input_names[0];
        if(!m->output_names.empty()) out_name = m->output_names[0];
    }
    static void crop_to_chw(const Mat &crop, int W, int H, float *outptr){
        Mat r; resize(crop, r, Size(W,H));
        Mat f; r.convertTo(f, CV_32F, 1.0f/255.0f);
        int idx=0;
        for(int c=0;c<3;c++) for(int y=0;y<H;y++) for(int x=0;x<W;x++){
            Vec3f v = f.at<Vec3f>(y,x);
            outptr[idx++] = v[c];
        }
    }
    vector<vector<float>> compute_embeddings(const vector<Mat> &crops){
        vector<vector<float>> embs;
        if(!m || crops.empty()) return embs;
        size_t N = crops.size();
        size_t D_in = size_t(3 * crop_h * crop_w);
        vector<float> batch(N * D_in);
        for(size_t i=0;i<N;++i){
            if(crops[i].empty()) continue;
            crop_to_chw(crops[i], crop_w, crop_h, batch.data() + i*D_in);
        }
        array<int64_t,4> dims = {(int64_t)N, 3, crop_h, crop_w};
        Ort::MemoryInfo meminfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        Ort::Value in_tensor = Ort::Value::CreateTensor<float>(meminfo, batch.data(), batch.size(), dims.data(), dims.size());
        const char* in_names[] = { in_name.c_str() };
        const char* out_names[] = { out_name.c_str() };
        vector<Ort::Value> outputs = m->session.Run(Ort::RunOptions{nullptr}, in_names, &in_tensor, 1, out_names, 1);
        if(outputs.empty()) return embs;
        Ort::Value &outv = outputs[0];
        auto info = outv.GetTensorTypeAndShapeInfo();
        vector<int64_t> shape = info.GetShape();
        if(shape.size()!=2) return embs;
        int D = (int)shape[1];
        float *outptr = outv.GetTensorMutableData<float>();
        embs.assign(N, vector<float>(D));
        for(int i=0;i<N;i++){
            double norm=0;
            for(int d=0; d<D; ++d){ float v = outptr[i*D + d]; embs[i][d]=v; norm += v*v; }
            norm = sqrt(norm) + 1e-9;
            for(int d=0; d<D; ++d) embs[i][d] = float(embs[i][d]/norm);
        }
        return embs;
    }
};

// ---------- Tracker ----------
struct MyTracker {
    SimpleKalman kf; int id; int age; Rect2f last_box; vector<float> embedding;
    MyTracker(int id_, const Rect2f &box, const vector<float> &emb) : kf(box.x + box.width*0.5f, box.y + box.height*0.5f), id(id_), age(0), last_box(box), embedding(emb) {}
};

// ---------- MAIN ----------
int main(int argc, char** argv){
    
    if(argc < 4){ cerr << "Usage: deepsort_serial_full <video_or_folder> <yolo.onnx> <reid.onnx> [max_frames]\n"; return 0; }
    string input_path = argv[1], yolo_p = argv[2], reid_p = argv[3];
    int max_frames = -1; if(argc >= 5) max_frames = stoi(argv[4]);

    fs::create_directories("output");
    auto global_start = chrono::steady_clock::now();

    // load models ONCE
    ONNXModel yolo_model(yolo_p);
    ONNXModel reid_model(reid_p);
    YoloONNX yolo(&yolo_model,  416, 416, 0.5f, 0.45f);
    ReidONNX reid(&reid_model, 128, 256);

    // build list of videos
    vector<fs::path> video_files;
    try {
        if(fs::is_regular_file(input_path)) {
            video_files.push_back(fs::path(input_path));
        } else if(fs::is_directory(input_path)) {
            for(auto &p : fs::directory_iterator(input_path)) {
                if(!p.is_regular_file()) continue;
                auto ext = p.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if(ext == ".mp4" || ext == ".avi" || ext == ".mov" || ext == ".mkv") video_files.push_back(p.path());
            }
            // sort for deterministic order
            sort(video_files.begin(), video_files.end());
        } else {
            cerr << "Input path is neither file nor directory: " << input_path << "\n";
            return -1;
        }
    } catch(const std::exception &e){
        cerr << "Filesystem error: " << e.what() << "\n";
        return -1;
    }

    if(video_files.empty()){
        cerr << "No video files found at: " << input_path << "\n";
        return -1;
    }

    // Globals for aggregated summary when multiple videos
    double global_det_sum = 0.0;
    size_t global_det_count = 0;
    double global_frame_sum = 0.0;
    size_t global_frame_count = 0;
    int global_processed_frames = 0;

    bool single_input = (video_files.size() == 1);

    // Process each video
    for(const auto &vid_path : video_files){
        cout << "Processing: " << vid_path << "\n";
            auto video_start = chrono::steady_clock::now();  // start timer for this video
        VideoCapture cap(vid_path.string());
        if(!cap.isOpened()){ cerr << "Cannot open video: " << vid_path << "\n"; continue; }

        int frameW = int(cap.get(CAP_PROP_FRAME_WIDTH));
        int frameH = int(cap.get(CAP_PROP_FRAME_HEIGHT));
        int total_frames = int(cap.get(CAP_PROP_FRAME_COUNT));
        double fps = cap.get(CAP_PROP_FPS); if(fps <= 0) fps = 25.0;
        if(max_frames > 0) total_frames = min(total_frames, max_frames);

        string base = vid_path.stem().string();
        string out_video = (fs::path("output") / ("serial_out_" + base + ".mp4")).string();
        VideoWriter writer(out_video, VideoWriter::fourcc('m','p','4','v'), fps, Size(frameW, frameH), true);

        ofstream dbg((fs::path("output") / ("debug_log_" + base + ".csv")).string());
        dbg << "frame,id,x,y,w,h,conf,matched,age,emb_cost,mot_cost,total_cost,warning\n";

        vector<MyTracker> trackers; int next_id = 0;
        vector<string> events;
        vector<double> det_times, frame_times;
        double alpha = 0.6;
        int max_age = 15;
        float cost_thresh = 0.7f;

        int processed = 0;
        for(int fidx=0; fidx < total_frames; ++fidx){
            Mat frame; if(!cap.read(frame)) break;
            auto t0 = chrono::steady_clock::now();

            // 1) detect
            auto tdet0 = chrono::steady_clock::now();
            auto dets = yolo.detect(frame);
            auto tdet1 = chrono::steady_clock::now();
            double det_time = chrono::duration<double>(tdet1 - tdet0).count();
            det_times.push_back(det_time);

            vector<Rect2f> boxes; vector<float> confs;
            for(auto &d : dets){ boxes.push_back(get<0>(d)); confs.push_back(get<1>(d)); }

            // 2) crops for ReID
            vector<Mat> crops; crops.reserve(boxes.size());
            for(auto &b : boxes){
                int x1 = max(0, (int)floor(b.x));
                int y1 = max(0, (int)floor(b.y));
                int x2 = min(frame.cols-1, (int)ceil(b.x + b.width));
                int y2 = min(frame.rows-1, (int)ceil(b.y + b.height));
                if(x2 <= x1 || y2 <= y1) { crops.emplace_back(); continue; }
                Mat crop = frame(Range(y1, y2+1), Range(x1, x2+1)).clone();
                cvtColor(crop, crop, COLOR_BGR2RGB);
                crops.push_back(crop);
            }
            auto embeddings = reid.compute_embeddings(crops);

            // 3) predict trackers
            for(auto &tr : trackers) tr.kf.predict(1.0f);

            // 4) cost matrix
            int m = (int)trackers.size(), n = (int)boxes.size();
            vector<vector<float>> cost;
            if(m>0 && n>0){
                cost.assign(m, vector<float>(n, 1.0f));
                for(int i=0;i<m;i++){
                    for(int j=0;j<n;j++){
                        float emb_cost = 1.0f;
                        if(!trackers[i].embedding.empty() && j < (int)embeddings.size() && !embeddings[j].empty())
                            emb_cost = cosine_distance(trackers[i].embedding, embeddings[j]);
                        float mot_cost = 1.0f - iou_rect(trackers[i].last_box, boxes[j]);
                        cost[i][j] = float(alpha*emb_cost + (1.0 - alpha)*mot_cost);
                    }
                }
            }
            vector<pair<int,int>> assignments;
            if(m>0 && n>0) assignments = hungarian_assign(cost, cost_thresh);

            // 5) update trackers
            vector<char> det_assigned(n, 0);
            vector<MyTracker> new_trackers;
            vector<int> track_to_det(m, -1);
            for(auto &p : assignments) if(p.first < m && p.second < n) track_to_det[p.first] = p.second;

            for(int i=0;i<m;i++){
                if(track_to_det[i] != -1){
                    int d = track_to_det[i];
                    float cx = boxes[d].x + boxes[d].width*0.5f;
                    float cy = boxes[d].y + boxes[d].height*0.5f;
                    trackers[i].kf.update(cx, cy);
                    trackers[i].last_box = boxes[d];
                    if(d < (int)embeddings.size() && !embeddings[d].empty()) trackers[i].embedding = embeddings[d];
                    trackers[i].age = 0;
                    new_trackers.push_back(trackers[i]);
                    det_assigned[d] = 1;

                    float emb_cost = (!trackers[i].embedding.empty() && d < (int)embeddings.size() && !embeddings[d].empty()) ? cosine_distance(trackers[i].embedding, embeddings[d]) : -1;
                    float mot_cost = 1.0f - iou_rect(trackers[i].last_box, boxes[d]);
                    float total_cost = (emb_cost >= 0) ? (alpha * emb_cost + (1.0 - alpha) * mot_cost) : mot_cost;
                    dbg << fidx << "," << trackers[i].id << "," << boxes[d].x << "," << boxes[d].y << "," << boxes[d].width << "," << boxes[d].height
                        << "," << confs[d] << ",1," << trackers[i].age << "," << emb_cost << "," << mot_cost << "," << total_cost << ",\n";
                } else {
                    trackers[i].age++;
                    if(trackers[i].age <= max_age) {
                        new_trackers.push_back(trackers[i]);
                        dbg << fidx << "," << trackers[i].id << "," << trackers[i].last_box.x << "," << trackers[i].last_box.y << "," << trackers[i].last_box.width << "," << trackers[i].last_box.height
                            << ",0,0," << trackers[i].age << ",-1,-1,-1,unmatched\n";
                    } else {
                        stringstream ss; ss << fidx << ",disappearance," << trackers[i].id << "," << trackers[i].last_box.x << "," << trackers[i].last_box.y << "," << trackers[i].last_box.width << "," << trackers[i].last_box.height;
                        events.push_back(ss.str());
                    }
                }
            }
            for(int j=0;j<n;j++){
                if(det_assigned[j]) continue;
                if(j >= (int)embeddings.size() || embeddings[j].empty()){
                    dbg << fidx << ",-1," << boxes[j].x << "," << boxes[j].y << "," << boxes[j].width << "," << boxes[j].height
                        << "," << confs[j] << ",0,0,-1,-1,-1,embedding_missing\n";
                    continue;
                }
                int id = next_id++;
                MyTracker tr(id, boxes[j], embeddings[j]);
                new_trackers.push_back(tr);
                stringstream ss; ss << fidx << ",appearance," << id << "," << boxes[j].x << "," << boxes[j].y << "," << boxes[j].width << "," << boxes[j].height;
                events.push_back(ss.str());
                dbg << fidx << "," << id << "," << boxes[j].x << "," << boxes[j].y << "," << boxes[j].width << "," << boxes[j].height
                    << "," << confs[j] << ",0,0,-1,-1,-1,new_tracker\n";
            }
            trackers.swap(new_trackers);

            // 6) draw
            Mat vis = frame.clone();
            for(auto &tr : trackers){
                Rect r((int)tr.last_box.x, (int)tr.last_box.y, (int)tr.last_box.width, (int)tr.last_box.height);
                Scalar color((tr.id*37)%255, (tr.id*17)%255, (tr.id*29)%255);
                rectangle(vis, r, color, 2);
                putText(vis, format("ID %d", tr.id), Point(max(0,r.x), max(0,r.y-5)), FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
            }
            if(writer.isOpened()) writer.write(vis);
            imshow("Debug Frame", vis);
            if(waitKey(1) == 'q') break;

            auto t1 = chrono::steady_clock::now();
            double frame_time = chrono::duration<double>(t1 - t0).count();
            frame_times.push_back(frame_time);

            processed++;
            if(max_frames>0 && processed >= max_frames) break;
            if(processed % 50 == 0) cout << "[frame " << fidx << "] det=" << det_time << "s total=" << frame_time << "s trackers=" << trackers.size() << "\n";
        }

        if(writer.isOpened()) writer.release();
        cap.release();
        dbg.close();

        // save events
        ofstream ofs((fs::path("output") / ("serial_events_" + base + ".csv")).string());
        ofs << "frame,event,id,x,y,w,h\n";
        for(auto &e : events) ofs << e << "\n";
        ofs.close();

        auto video_end = chrono::steady_clock::now();
        double wall_time = chrono::duration<double>(video_end - video_start).count();  // wall-clock duration

        // timings for this video
        double total_time = 0.0; // compute total execution time per video as sum of frame times
        total_time = accumulate(frame_times.begin(), frame_times.end(), 0.0);
        double avg_det = det_times.empty() ? 0.0 :
            accumulate(det_times.begin(), det_times.end(), 0.0) / det_times.size();
        double avg_frame = frame_times.empty() ? 0.0 :
            accumulate(frame_times.begin(), frame_times.end(), 0.0) / frame_times.size();
        double approx_fps = total_time > 0 ? (double)processed / total_time : 0.0;

        // write per-video timings file
        ofstream ofs2((fs::path("output") / ("timings_" + base + ".txt")).string());
        ofs2 << "Frames processed: " << processed << "\n";
        ofs2 << "Total execution time (s): " << total_time << "\n";
        ofs2 << "Wall-clock time (s): " << wall_time << " (includes decoding, waits, etc)\n";
        ofs2 << "Average detection time (s): " << avg_det << "\n";
        ofs2 << "Average total frame time (s): " << avg_frame << "\n";
        ofs2 << "Approx FPS (serial): " << approx_fps << "\n";
        ofs2.close();

        // console output per-video (if single input) or short per-video notice (if multi)
        if(single_input){
            cout << "=== SUMMARY ===\n";
            cout << "Frames processed: " << processed << "\n";
            cout << "Total execution time (s): " << total_time << "\n";
            cout << "Average detection time (s): " << avg_det << "\n";
            cout << "Average total frame time (s): " << avg_frame << "\n";
            cout << "Approx FPS (serial): " << approx_fps << "\n";
            cout << "Saved video to " << out_video << ", logs in output/\n";
        } else {
            cout << "Finished " << base << ": frames=" << processed << " total_time(s)=" << total_time << " fps=" << approx_fps << "\n";
        }

        // accumulate to global stats
        global_processed_frames += processed;
        global_det_sum += accumulate(det_times.begin(), det_times.end(), 0.0);
        global_det_count += det_times.size();
        global_frame_sum += accumulate(frame_times.begin(), frame_times.end(), 0.0);
        global_frame_count += frame_times.size();
    } // end for each video

    // global summary if multiple videos
    auto global_end = chrono::steady_clock::now();
    double global_total_time = chrono::duration<double>(global_end - global_start).count();
    if(!single_input){
        double overall_avg_det = (global_det_count==0) ? 0.0 : (global_det_sum / (double)global_det_count);
        double overall_avg_frame = (global_frame_count==0) ? 0.0 : (global_frame_sum / (double)global_frame_count);
        double overall_fps = global_total_time > 0 ? (double)global_processed_frames / global_total_time : 0.0;

        cout << "\n=== AGGREGATED SUMMARY ===\n";
        cout << "Videos processed: " << video_files.size() << "\n";
        cout << "Total frames processed: " << global_processed_frames << "\n";
        cout << "Total wall-clock time (s): " << global_total_time << "\n";
        cout << "Average detection time (s) [avg over frames]: " << overall_avg_det << "\n";
        cout << "Average total frame time (s) [avg over frames]: " << overall_avg_frame << "\n";
        cout << "Overall Approx FPS (serial): " << overall_fps << "\n";
        cout << "Per-video outputs and logs saved in output/\n";
    }

    return 0;
}
