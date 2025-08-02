from pathlib import Path
import numpy as np

import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork, GatherData

from utils.arguments import initialize_argparser
from utils.process_keypoints import LandmarksProcessing
from utils.node_creators import create_crop_node
from utils.annotation_node import AnnotationNode
from utils.host_concatenate_head_pose import ConcatenateHeadPose

DET_MODEL = "luxonis/yunet:320x240"
HEAD_POSE_MODEL = "luxonis/head-pose-estimation:60x60"
GAZE_MODEL = "luxonis/gaze-estimation-adas:60x60"
REQ_WIDTH, REQ_HEIGHT = (640, 480)

_, args = initialize_argparser()

visualizer = dai.RemoteConnection(httpPort=8082)
pipeline = dai.Pipeline()

platform = "RVC2"  # デフォルト値
print(f"Platform: {platform}")

if platform == "RVC4":
    DET_MODEL = "luxonis/scrfd-face-detection:10g-640x640"
    REQ_WIDTH, REQ_HEIGHT = (768, 768)

frame_type = (
    dai.ImgFrame.Type.BGR888i if platform == "RVC4" else dai.ImgFrame.Type.BGR888p
)

if args.fps_limit is None:
    args.fps_limit = 15 if platform == "RVC2" else 30
    print(f"\nFPS limit set to {args.fps_limit} for {platform} platform.\n")

print("Creating pipeline...")

# face detection model
det_model_description = dai.NNModelDescription(DET_MODEL)
det_model_description.platform = platform
det_model_nn_archive = dai.NNArchive(dai.getModelFromZoo(det_model_description))

# head pose model
head_pose_model_description = dai.NNModelDescription(HEAD_POSE_MODEL)
head_pose_model_description.platform = platform
head_pose_model_nn_archive = dai.NNArchive(
    dai.getModelFromZoo(head_pose_model_description)
)

# gaze estimation model
gaze_model_description = dai.NNModelDescription(GAZE_MODEL)
gaze_model_description.platform = platform
gaze_model_nn_archive = dai.NNArchive(dai.getModelFromZoo(gaze_model_description))

# media/camera input
if args.media_path:
    replay = pipeline.create(dai.node.ReplayVideo)
    replay.setReplayVideoFile(Path(args.media_path))
    replay.setOutFrameType(frame_type)
    replay.setLoop(True)
    if args.fps_limit:
        replay.setFps(args.fps_limit)
    replay.setSize(REQ_WIDTH, REQ_HEIGHT)
else:
    cam = pipeline.create(dai.node.Camera).build()
    cam_out = cam.requestOutput(
        size=(REQ_WIDTH, REQ_HEIGHT), type=frame_type, fps=args.fps_limit
    )
input_node_out = replay.out if args.media_path else cam_out

# resize to det model input size
resize_node = pipeline.create(dai.node.ImageManip)
resize_node.initialConfig.setOutputSize(
    det_model_nn_archive.getInputWidth(), det_model_nn_archive.getInputHeight()
)
resize_node.setMaxOutputFrameSize(
    det_model_nn_archive.getInputWidth() * det_model_nn_archive.getInputHeight() * 3
)
resize_node.initialConfig.setReusePreviousImage(False)
resize_node.inputImage.setBlocking(True)
input_node_out.link(resize_node.inputImage)

det_nn: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
    resize_node.out, det_model_nn_archive
)
det_nn.input.setBlocking(True)

# detection processing
detection_process_node = pipeline.create(LandmarksProcessing)
detection_process_node.set_source_size(REQ_WIDTH, REQ_HEIGHT)
detection_process_node.set_target_size(
    head_pose_model_nn_archive.getInputWidth(),
    head_pose_model_nn_archive.getInputHeight(),
)
det_nn.out.link(detection_process_node.detections_input)

left_eye_crop_node = create_crop_node(
    pipeline, input_node_out, detection_process_node.left_config_output
)
right_eye_crop_node = create_crop_node(
    pipeline, input_node_out, detection_process_node.right_config_output
)
face_crop_node = create_crop_node(
    pipeline, input_node_out, detection_process_node.face_config_output
)

# head pose estimation
head_pose_nn = pipeline.create(ParsingNeuralNetwork).build(
    face_crop_node.out, head_pose_model_nn_archive
)
head_pose_nn.input.setBlocking(True)

head_pose_concatenate_node = pipeline.create(ConcatenateHeadPose).build(
    head_pose_nn.getOutput(0), head_pose_nn.getOutput(1), head_pose_nn.getOutput(2)
)

# gaze estimation
gaze_estimation_node = pipeline.create(dai.node.NeuralNetwork)
gaze_estimation_node.setNNArchive(gaze_model_nn_archive)
head_pose_concatenate_node.output.link(
    gaze_estimation_node.inputs["head_pose_angles_yaw_pitch_roll"]
)
left_eye_crop_node.out.link(gaze_estimation_node.inputs["left_eye_image"])
right_eye_crop_node.out.link(gaze_estimation_node.inputs["right_eye_image"])
gaze_estimation_node.inputs["head_pose_angles_yaw_pitch_roll"].setBlocking(True)
gaze_estimation_node.inputs["left_eye_image"].setBlocking(True)
gaze_estimation_node.inputs["right_eye_image"].setBlocking(True)
gaze_estimation_node.inputs["left_eye_image"].setMaxSize(5)
gaze_estimation_node.inputs["right_eye_image"].setMaxSize(5)
gaze_estimation_node.inputs["head_pose_angles_yaw_pitch_roll"].setMaxSize(5)

# detections and gaze estimations sync
gather_data_node = pipeline.create(GatherData).build(args.fps_limit)
gaze_estimation_node.out.link(gather_data_node.input_data)
det_nn.out.link(gather_data_node.input_reference)

# annotation
annotation_node = pipeline.create(AnnotationNode).build(gather_data_node.out)

# 視線データ取得用のキュー
gaze_raw_queue = gaze_estimation_node.out.createOutputQueue()

# visualization
visualizer.addTopic("Video", input_node_out, "images")
visualizer.addTopic("Gaze", annotation_node.out, "images")

print("Pipeline created.")
pipeline.start()
visualizer.registerPipeline(pipeline)

print("視線成分表示開始...")
print("画面確認: http://localhost:8082")
print("X(左右) | Y(上下)")
print("-" * 20)

try:
    while pipeline.isRunning():
        if gaze_raw_queue.has():
            gaze_raw_data = gaze_raw_queue.get()
            
            try:
                # 視線データを取得
                if hasattr(gaze_raw_data, 'getFirstLayerFp16'):
                    raw_gaze = gaze_raw_data.getFirstLayerFp16()
                elif hasattr(gaze_raw_data, 'getData'):
                    raw_gaze = gaze_raw_data.getData()
                else:
                    raw_gaze = []
                
                # 視線成分を表示
                if len(raw_gaze) >= 2:
                    x_component = raw_gaze[0]  # 左右
                    y_component = raw_gaze[1]  # 上下
                    print(f"{x_component:6.3f} | {y_component:6.3f}")
                
            except Exception as e:
                print(f"データ取得エラー: {e}")
        
        # 終了チェック
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("\n終了します...")
            break

except KeyboardInterrupt:
    print("\n終了します...")
finally:
    pipeline.stop()