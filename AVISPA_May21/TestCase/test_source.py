import argparse
import os
import glob


parser = argparse.ArgumentParser()
parser.add_argument("--source", required=True, help='Can be a directory with images, a path to a single image,'
                                                    ' video file or "webcam"')
parser.add_argument("--expected")
parser.add_argument("--export_mistakes", action="store_true")
parser.add_argument("--export_cropped", action="store_true")
parser.add_argument("--export_raw", action="store_true")
parser.add_argument("--show_detected", action="store_true")
parser.add_argument("--show_classified", action="store_true")
parser.add_argument("--show_cropped", action="store_true")
parser.add_argument("--show_raw", action="store_true")
parser.add_argument("--wait", action="store_true")
parser.add_argument("--cam", default=0, type=int)
parser.add_argument("--limit", default=0, type=int)
parser.add_argument("--skip", default=0, type=int)
parser.add_argument("--sharpness", default=0, type=int)
parser.add_argument("--log_level", default="info", choices=["info", "debug"])
parser.add_argument("--skip_detection", action="store_true")
parser.add_argument("--skip_classification", action="store_true")
parser.add_argument("--classification_model")
parser.add_argument("--classification_model_name")
parser.add_argument("--classification_score_threshold")
parser.add_argument("--sharpness_percentile")
args = parser.parse_args()


if args.source == "webcam":
	process_webcam(args.cam)
	evaluate.print_summary(args)
elif os.path.isfile(args.source):
	process_video(args.source)
	evaluate.print_summary(args)
elif os.path.isdir(args.source):
	process_directory(args.source)
else:
	logging.error("Invalid source")
	exit()