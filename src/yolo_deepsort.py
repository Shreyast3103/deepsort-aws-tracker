import cv2
import numpy as np
import onnxruntime as ort
import os
import sys
import time
import math
import csv
from pathlib import Path
from collections import defaultdict

# ==============================
# Simple Kalman (same as C++)
# ==============================
class SimpleKalman:
    def __init__(self, cx=0.0, cy=0.0):
        self.x = np.zeros((4,1), dtype=np.float32)
        self.x[0,0] = cx
        self.x[1,0] = cy
        self.P = np.eye(4, dtype=np.float32) * 500.0
        self.Q = np.eye(4, dtype=np.float32) * 1.0
        self.R = np.eye(2, dtype=np.float32) * 10.0

    def predict(self, dt=1.0):
        F = np.eye(4, dtype=np.float32)
        F[0,2] = dt
        F[1,3] = dt
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

    def update(self, mx, my):
        H = np.zeros((2,4), dtype=np.float32)
        H[0,0] = 1
        H[1,1] = 1
        z = np.array([[mx],[my]], dtype=np.float32)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - H @ self.x)
        I = np.eye(4, dtype=np.float32)
        self.P = (I - K @ H) @ self.P

# ==============================
# Utilities
# ==============================
def iou_rect(A, B):
    xA = max(A[0], B[0])
    yA = max(A[1], B[1])
    xB = min(A[0]+A[2], B[0]+B[2])
    yB = min(A[1]+A[3], B[1]+B[3])
    w = max(0.0, xB-xA)
    h = max(0.0, yB-yA)
    inter = w*h
    ua = A[2]*A[3] + B[2]*B[3] - inter + 1e-9
    return inter/ua

def cosine_distance(a,b):
    if len(a)==0 or len(b)==0:
        return 1.0
    dot = np.dot(a,b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na<1e-9 or nb<1e-9:
        return 1.0
    return 1.0 - dot/(na*nb+1e-9)

# ==============================
# Hungarian (Manual — same logic)
# ==============================
def hungarian_assign(cost, cost_thresh=0.7):
    if len(cost)==0:
        return []
    from scipy.optimize import linear_sum_assignment
    cost_matrix = np.array(cost)
    rows, cols = linear_sum_assignment(cost_matrix)
    res = []
    for r,c in zip(rows,cols):
        if cost_matrix[r,c] <= cost_thresh:
            res.append((r,c))
    return res

# ==============================
# YOLO ONNX
# ==============================
class YoloONNX:
    def __init__(self, model_path, input_w=416, input_h=416,
                 conf_thresh=0.5, nms_thresh=0.45):
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_w = input_w
        self.input_h = input_h
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

    def preprocess(self, frame):
        img = cv2.resize(frame,(self.input_w,self.input_h))
        img = img.astype(np.float32)/255.0
        img = np.transpose(img,(2,0,1))
        img = np.expand_dims(img,0)
        return img

    def detect(self, frame):
        inp = self.preprocess(frame)
        outputs = self.session.run([self.output_name], {self.input_name: inp})
        out = outputs[0]
        if len(out.shape)==3:
            out = out[0]
        boxes=[]
        scores=[]
        for row in out:
            if len(row)<6:
                continue
            x1,y1,x2,y2,conf,cls = row[:6]
            if conf < self.conf_thresh:
                continue
            if int(round(cls))!=0:
                continue
            sx = frame.shape[1]/self.input_w
            sy = frame.shape[0]/self.input_h
            x1*=sx; x2*=sx; y1*=sy; y2*=sy
            w = max(0,x2-x1)
            h = max(0,y2-y1)
            boxes.append([x1,y1,w,h])
            scores.append(conf)
        # simple NMS
        idx = np.argsort(scores)[::-1]
        keep=[]
        removed=[False]*len(boxes)
        for i in idx:
            if removed[i]: continue
            keep.append(i)
            for j in idx:
                if i==j or removed[j]: continue
                if iou_rect(boxes[i],boxes[j])>self.nms_thresh:
                    removed[j]=True
        dets=[]
        for i in keep:
            dets.append((boxes[i],scores[i],0))
        return dets

# ==============================
# ReID ONNX
# ==============================
class ReidONNX:
    def __init__(self, model_path, w=128, h=256):
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.w=w; self.h=h

    def compute_embeddings(self,crops):
        if len(crops)==0:
            return []
        batch=[]
        for crop in crops:
            if crop is None or crop.size==0:
                batch.append(np.zeros((3,self.h,self.w),dtype=np.float32))
                continue
            crop=cv2.resize(crop,(self.w,self.h))
            crop=crop.astype(np.float32)/255.0
            crop=np.transpose(crop,(2,0,1))
            batch.append(crop)
        batch=np.array(batch,dtype=np.float32)
        outputs=self.session.run([self.output_name],{self.input_name:batch})
        embs=outputs[0]
        for i in range(len(embs)):
            norm=np.linalg.norm(embs[i])+1e-9
            embs[i]=embs[i]/norm
        return embs

# ==============================
# Tracker
# ==============================
class MyTracker:
    def __init__(self,id_,box,emb):
        cx=box[0]+box[2]*0.5
        cy=box[1]+box[3]*0.5
        self.kf=SimpleKalman(cx,cy)
        self.id=id_
        self.age=0
        self.last_box=box
        self.embedding=emb

# ==============================
# MAIN
# ==============================
def main():
    if len(sys.argv)<4:
        print("Usage: deepsort_serial_full <video_or_folder> <yolo.onnx> <reid.onnx> [max_frames]")
        return

    input_path=sys.argv[1]
    yolo_p=sys.argv[2]
    reid_p=sys.argv[3]
    max_frames=int(sys.argv[4]) if len(sys.argv)>=5 else -1

    Path("output").mkdir(exist_ok=True)

    global_start=time.time()

    yolo=YoloONNX(yolo_p)
    reid=ReidONNX(reid_p)

    video_files=[]
    p=Path(input_path)
    if p.is_file():
        video_files.append(p)
    elif p.is_dir():
        for f in p.iterdir():
            if f.suffix.lower() in [".mp4",".avi",".mov",".mkv"]:
                video_files.append(f)
        video_files.sort()
    else:
        print("Input path is neither file nor directory:",input_path)
        return

    if len(video_files)==0:
        print("No video files found at:",input_path)
        return

    global_processed_frames=0
    global_frame_sum=0
    global_det_sum=0
    global_det_count=0

    single_input=(len(video_files)==1)

    for vid_path in video_files:
        print("Processing:",vid_path)
        video_start=time.time()

        cap=cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            print("Cannot open video:",vid_path)
            continue

        frameW=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameH=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps=cap.get(cv2.CAP_PROP_FPS)
        if fps<=0: fps=25.0
        if max_frames>0:
            total_frames=min(total_frames,max_frames)

        base=vid_path.stem
        out_video=str(Path("output")/f"serial_out_{base}.mp4")
        writer=cv2.VideoWriter(out_video,
                               cv2.VideoWriter_fourcc(*'mp4v'),
                               fps,(frameW,frameH))

        dbg=open(str(Path("output")/f"debug_log_{base}.csv"),"w",newline="")
        dbg_writer=csv.writer(dbg)
        dbg_writer.writerow(["frame","id","x","y","w","h","conf","matched","age","emb_cost","mot_cost","total_cost","warning"])

        trackers=[]
        next_id=0
        events=[]
        det_times=[]
        frame_times=[]

        alpha=0.6
        max_age=15
        cost_thresh=0.7

        processed=0

        for fidx in range(total_frames):
            ret,frame=cap.read()
            if not ret:
                break

            t0=time.time()

            tdet0=time.time()
            dets=yolo.detect(frame)
            tdet1=time.time()
            det_time=tdet1-tdet0
            det_times.append(det_time)

            boxes=[d[0] for d in dets]
            confs=[d[1] for d in dets]

            crops=[]
            for b in boxes:
                x1=int(max(0,b[0]))
                y1=int(max(0,b[1]))
                x2=int(min(frame.shape[1]-1,b[0]+b[2]))
                y2=int(min(frame.shape[0]-1,b[1]+b[3]))
                if x2<=x1 or y2<=y1:
                    crops.append(None)
                else:
                    crop=frame[y1:y2,x1:x2].copy()
                    crop=cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)
                    crops.append(crop)

            embeddings=reid.compute_embeddings(crops)

            for tr in trackers:
                tr.kf.predict(1.0)

            m=len(trackers)
            n=len(boxes)
            cost=[]
            if m>0 and n>0:
                for i in range(m):
                    row=[]
                    for j in range(n):
                        emb_cost=cosine_distance(trackers[i].embedding,embeddings[j])
                        mot_cost=1.0-iou_rect(trackers[i].last_box,boxes[j])
                        row.append(alpha*emb_cost+(1-alpha)*mot_cost)
                    cost.append(row)

            assignments=[]
            if m>0 and n>0:
                assignments=hungarian_assign(cost,cost_thresh)

            det_assigned=[False]*n
            new_trackers=[]
            track_to_det=[-1]*m
            for r,c in assignments:
                if r<m and c<n:
                    track_to_det[r]=c

            for i in range(m):
                if track_to_det[i]!=-1:
                    d=track_to_det[i]
                    cx=boxes[d][0]+boxes[d][2]*0.5
                    cy=boxes[d][1]+boxes[d][3]*0.5
                    trackers[i].kf.update(cx,cy)
                    trackers[i].last_box=boxes[d]
                    trackers[i].embedding=embeddings[d]
                    trackers[i].age=0
                    new_trackers.append(trackers[i])
                    det_assigned[d]=True

                    dbg_writer.writerow([fidx,trackers[i].id,
                                         boxes[d][0],boxes[d][1],boxes[d][2],boxes[d][3],
                                         confs[d],1,trackers[i].age,"","","",""])
                else:
                    trackers[i].age+=1
                    if trackers[i].age<=max_age:
                        new_trackers.append(trackers[i])
                        dbg_writer.writerow([fidx,trackers[i].id,
                                             trackers[i].last_box[0],trackers[i].last_box[1],
                                             trackers[i].last_box[2],trackers[i].last_box[3],
                                             0,0,trackers[i].age,"","","","unmatched"])
                    else:
                        events.append(f"{fidx},disappearance,{trackers[i].id},"
                                      f"{trackers[i].last_box[0]},{trackers[i].last_box[1]},"
                                      f"{trackers[i].last_box[2]},{trackers[i].last_box[3]}")

            for j in range(n):
                if det_assigned[j]:
                    continue
                id_=next_id
                next_id+=1
                tr=MyTracker(id_,boxes[j],embeddings[j])
                new_trackers.append(tr)
                events.append(f"{fidx},appearance,{id_},{boxes[j][0]},{boxes[j][1]},"
                              f"{boxes[j][2]},{boxes[j][3]}")
                dbg_writer.writerow([fidx,id_,
                                     boxes[j][0],boxes[j][1],boxes[j][2],boxes[j][3],
                                     confs[j],0,0,"","","","new_tracker"])

            trackers=new_trackers

            vis=frame.copy()
            for tr in trackers:
                x,y,w,h=map(int,tr.last_box)
                color=((tr.id*37)%255,(tr.id*17)%255,(tr.id*29)%255)
                cv2.rectangle(vis,(x,y),(x+w,y+h),color,2)
                cv2.putText(vis,f"ID {tr.id}",(x,max(0,y-5)),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

            writer.write(vis)
            DISPLAY = False  # set True for local

            if DISPLAY:
                cv2.imshow("Debug Frame", vis)
                if cv2.waitKey(1) == ord('q'):
                    break

            frame_time=time.time()-t0
            frame_times.append(frame_time)

            processed+=1
            if max_frames>0 and processed>=max_frames:
                break
            if processed%50==0:
                print(f"[frame {fidx}] det={det_time}s total={frame_time}s trackers={len(trackers)}")

        writer.release()
        cap.release()
        dbg.close()

        with open(str(Path("output")/f"serial_events_{base}.csv"),"w") as f:
            f.write("frame,event,id,x,y,w,h\n")
            for e in events:
                f.write(e+"\n")

        video_end=time.time()
        wall_time=video_end-video_start

        total_time=sum(frame_times)
        avg_det=sum(det_times)/len(det_times) if det_times else 0
        avg_frame=sum(frame_times)/len(frame_times) if frame_times else 0
        approx_fps=processed/total_time if total_time>0 else 0

        with open(str(Path("output")/f"timings_{base}.txt"),"w") as f:
            f.write(f"Frames processed: {processed}\n")
            f.write(f"Total execution time (s): {total_time}\n")
            f.write(f"Wall-clock time (s): {wall_time}\n")
            f.write(f"Average detection time (s): {avg_det}\n")
            f.write(f"Average total frame time (s): {avg_frame}\n")
            f.write(f"Approx FPS (serial): {approx_fps}\n")

        if single_input:
            print("=== SUMMARY ===")
            print("Frames processed:",processed)
            print("Total execution time (s):",total_time)
            print("Average detection time (s):",avg_det)
            print("Average total frame time (s):",avg_frame)
            print("Approx FPS (serial):",approx_fps)
            print("Saved video to",out_video,", logs in output/")
        else:
            print(f"Finished {base}: frames={processed} total_time(s)={total_time} fps={approx_fps}")

        global_processed_frames+=processed
        global_frame_sum+=sum(frame_times)
        global_det_sum+=sum(det_times)
        global_det_count+=len(det_times)

    global_total_time=time.time()-global_start

    if not single_input:
        overall_avg_det=global_det_sum/global_det_count if global_det_count else 0
        overall_avg_frame=global_frame_sum/global_processed_frames if global_processed_frames else 0
        overall_fps=global_processed_frames/global_total_time if global_total_time>0 else 0

        print("\n=== AGGREGATED SUMMARY ===")
        print("Videos processed:",len(video_files))
        print("Total frames processed:",global_processed_frames)
        print("Total wall-clock time (s):",global_total_time)
        print("Average detection time (s) [avg over frames]:",overall_avg_det)
        print("Average total frame time (s) [avg over frames]:",overall_avg_frame)
        print("Overall Approx FPS (serial):",overall_fps)
        print("Per-video outputs and logs saved in output/")

if __name__=="__main__":
    main()
