import cv2
import time
import tensorflow as tf
from glob import glob
from detection_test_pb import detection
from recognition_test_pb import recognition
from text_recognition.model import TextRecognition
import os
import numpy as np
from tqdm import tqdm

def compose_video(frames_path, fps=1):
    output_video_path = './output_cv_1/detection_saturized.mp4'
    print(frames_path)
    # raise ValueError("stop")
    frame_files = sorted(glob(os.path.join(frames_path, '*.jpg')))
    print(frame_files)
    if not frame_files:
        print("No frames.")
        return

    frame = cv2.imread(frame_files[0])
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame_file in frame_files:
        video.write(cv2.imread(frame_file))

    video.release()
    print("Video complete on path: ", output_video_path)

def extract_frames(video_path, output_dir, fps_interval=1):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print("Error: Could not open video.")
        return

    fps = video.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * fps_interval)

    count = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{count // frame_interval}.jpg")
            cv2.imwrite(frame_path, frame)
            print(f"Frame saved: {frame_path}")

        count += 1
    video.release()
    print("Frame extraction complete.")

def containernumber_recognition(img_path, res_path):
    detection_model_path = "pb_models/detection.pb"
    recognition_model_h_path = "ckpt/recognition_h/model_all.ckpt-8000"
    recognition_model_v_path = "ckpt/recognition_v/model_all.ckpt-146000"

    with tf.Graph().as_default():
        detection_graph_def = tf.GraphDef()
        with open(detection_model_path, "rb") as f:
            detection_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(detection_graph_def, name="")

        sess_d=tf.Session()
        init = tf.global_variables_initializer()
        sess_d.run(init)
        input_x = sess_d.graph.get_tensor_by_name("Placeholder:0")
        segm_logits = sess_d.graph.get_tensor_by_name("model/segm_logits/add:0")
        link_logits = sess_d.graph.get_tensor_by_name("model/link_logits/Reshape:0")

    bs = 4
    model = TextRecognition(is_training=False, num_classes=37)

    images_ph_h = tf.placeholder(tf.float32, [bs, 32, 240, 1])
    model_out_h = model(inputdata=images_ph_h)
    saver_h = tf.train.Saver()
    sess_r_h=tf.Session()
    saver_h.restore(sess=sess_r_h, save_path=recognition_model_h_path)
    decoded_h, _ = tf.nn.ctc_beam_search_decoder(model_out_h, 60 * np.ones(bs), merge_repeated=False)

    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        images_ph_v = tf.placeholder(tf.float32, [bs, 32, 320, 1])
        model_out_v = model(inputdata=images_ph_v)
        saver_v = tf.train.Saver()
        sess_r_v=tf.Session()
        saver_v.restore(sess=sess_r_v, save_path=recognition_model_v_path)
        decoded_v, _ = tf.nn.ctc_beam_search_decoder(model_out_v, 80 * np.ones(bs), merge_repeated=False)
        
    res_txt = open('containernumber_result.txt', 'w')
    impaths = glob(os.path.join(img_path, '*'))
    res_dir = res_path
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    config = {}
    config['segm_conf_thr'] = 0.8
    config['link_conf_thr'] = 0.8
    config['min_area'] = 300
    config['min_height'] = 10

    total_time1 = time.time()  
    for impath in tqdm(impaths):
        
        imname = os.path.basename(impath)
        im = cv2.imread(impath)
        print(imname)
        t1 = time.time()
        
        bboxs = detection(im, sess_d, input_x, segm_logits, link_logits, config)
        
        for bbox in bboxs:
            pts = [int(p) for p in bbox.split(",")]
            cv2.rectangle(im, (pts[0], pts[1]), (pts[4], pts[5]), (0, 255, 0), 2)
            # Write bbox on result image
            # bbox_text = f"({pts[0]}, {pts[1]}) - ({pts[4]}, {pts[5]})"
            # cv2.putText(im, bbox_text, (pts[0], pts[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        t2 = time.time()

        try:
            predicted = recognition(im, sess_r_h, sess_r_v , bboxs, (240, 32), images_ph_h, images_ph_v, model_out_h, model_out_v, decoded_h, decoded_v)
        except ValueError as e:
            print(f"Error processing frame {impath}: {e}")
            continue

        cv2.putText(im, predicted, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        t3 = time.time()
        print('recognition_time: ', (t3-t2),'result', predicted)
        cv2.imwrite(os.path.join(res_dir, imname), im)
        line = imname + ' ' + predicted + '\n'
        res_txt.write(line)

    res_txt.close()
    total_time2 = time.time()
    print('total_time: ', (total_time2 - total_time1))

    # from accuracy import acc
    # acc('containernumber_result.txt')

def main():

    # using CPU
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    img_path = './samples'
    res_path = "./output_samples"
    containernumber_recognition(img_path, res_path)

    # Experiment with video as input
    # video_file_path = "/Users/audinamaharani/Downloads/256209FF-1308-4774-B617-A31B0EF1E9AF.mp4"
    # temp_path = "./temp_detection_saturized"
    # extract_frames(video_file_path, temp_path)
    # compose_video(temp_path)

if __name__ =="__main__":
    main()