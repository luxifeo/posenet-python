import cv2
import tensorflow as tf
import posenet
import time

cap = cv2.VideoCapture('slav')

with tf.Session() as sess:
    model_cfg, model_outputs = posenet.load_model(model_id=101, sess=sess)
    output_stride = model_cfg['output_stride']
    while(cap.isOpened()):
        start = time.time()
        ret, frame = cap.read()
        input_image, draw_image, output_scale = posenet.process_input(frame)
        heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
            model_outputs,
            feed_dict={'image:0': input_image}
        )
        pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
            heatmaps_result.squeeze(axis=0),
            offsets_result.squeeze(axis=0),
            displacement_fwd_result.squeeze(axis=0),
            displacement_bwd_result.squeeze(axis=0),
            output_stride=output_stride,
            max_pose_detections=10,
            min_pose_score=0.25)
        keypoint_coords *= output_scale
        draw_image = posenet.draw_skel_and_kp(
            draw_image, pose_scores, keypoint_scores, keypoint_coords,
            min_pose_score=0.25, min_part_score=0.25)
        end = time.time()
        fps = 'fps: {:.2f}'.format(1.0 / (end - start))
        cv2.putText(draw_image, fps, (15, 15), cv2.FONT_ITALIC,
                    0.5, (0, 0, 255), thickness=2)
        cv2.imshow('thicc', draw_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
