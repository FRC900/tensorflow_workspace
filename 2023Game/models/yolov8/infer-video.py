import argparse
import cv2
import torch

from sys import path
path.append('/home/ubuntu/YOLOv8-TensorRT')
from models import TRTModule  # isort:skip
path.append('/home/ubuntu/tensorflow_workspace/2023Game/models')
import timing

from config_frc2023 import OBJECT_CLASSES, COLORS
from models.torch_utils import det_postprocess
from models.utils import blob, letterbox, path_to_list


def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    Engine = TRTModule(args.engine, device)
    H, W = Engine.inp_info[0].shape[-2:]

    # set desired output names order
    Engine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])

    cap = cv2.VideoCapture(args.input_video)
    t = timing.Timings()

    while True:
        t.start('frame')
        t.start('vid')
        ret, bgr = cap.read()
        t.end('vid')
        if not ret:
            break
        t.start('cv')

        draw = bgr.copy()
        bgr, ratio, dwdh = letterbox(bgr, (W, H)) # resize while maintaining aspect ratio
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) # standard color conversion
        tensor = blob(rgb, return_seg=False) # convert to float, transpose, scale from 0.0->1.0
        dwdh = torch.asarray(dwdh * 2, dtype=torch.float32, device=device)
        tensor = torch.asarray(tensor, device=device)
        t.end('cv')
        # inference
        t.start('inference')
        data = Engine(tensor)
        t.end('inference')

        t.start('viz')
        bboxes, scores, labels = det_postprocess(data)
        if bboxes.numel() != 0:
            bboxes -= dwdh
            bboxes /= ratio

            for (bbox, score, label) in zip(bboxes, scores, labels):
                bbox = bbox.round().int().tolist()
                cls_id = int(label)
                cls = OBJECT_CLASSES.get_name(cls_id)
                color = COLORS[cls_id]
                cv2.rectangle(draw, bbox[:2], bbox[2:], color, 2)
                cv2.putText(draw,
                            f'{cls}:{score:.3f}', (bbox[0], bbox[1] - 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, [225, 255, 255],
                            thickness=2)

        if args.show:
            cv2.imshow('result', draw)
            t.end('viz')
            key = cv2.waitKey(1) & 0x000000ff
            if key == 27:
                break
        else:
            t.end('viz')

        t.end('frame')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, help='Engine file')
    parser.add_argument('--input-video', type=str, help='Input video')
    parser.add_argument('--show',
                        action='store_true',
                        help='Show the detection results')
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help='TensorRT infer device')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
