import argparse
import time
from SDD_FIQA import *
from animations.animations import *
from face_reg.detection import *
from SmileScore.smileScore import *
from animations.make_video import *
import datetime
from misc.log import *
from config import *
from misc.visualize import *
from misc.utils import *

import warnings
warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description='Face Detection and Recognition',
                                     usage='A module to detect and recognize faces in pictures')

    parser.add_argument('--anchor_dataset_path',
                        help='Path to your folder containing anchor images',
                        type=str,
                        required=True,
                        default=None)

    parser.add_argument('--original_dataset_path',
                        help='Path to your folder containing input images',
                        type=str,
                        required=True,
                        default=None)

    parser.add_argument('--output_path',
                        help='Output video path',
                        type=str,
                        required=True,
                        default=None)

    parser.add_argument('--number_of_images',
                        help='Number of images will be presented in the video',
                        type=int,
                        required=False,
                        default=6)

    parser.add_argument('--effect_speed',
                        help='Video args',
                        type=int,
                        required=False,
                        default=1)

    parser.add_argument('--duration',
                        help='Video args',
                        type=int,
                        required=False,
                        default=3)

    parser.add_argument('--fps',
                        help='Video args',
                        type=int,
                        required=False,
                        default=75)
                        
    parser.add_argument('--fraction',
                        help='Resize images',
                        type=float,
                        required=False,
                        default=1)

    parser.add_argument('--find_person',
                        help='Find the person',
                        type = str,
                        required=False,
                        default=None)

    parser.add_argument('--find_all',
                        help='Only keep images that have all people in the query search',
                        type=bool,
                        required=False,
                        default=False)

    parser.add_argument('--log',
                        help='Find the person',
                        type=bool,
                        required=False,
                        default=False)

    parser.add_argument('--visualize_boxes',
                        help='Visualizing bounding boxes in the video',
                        type=bool,
                        required=False,
                        default=False)

    args = parser.parse_args()
    return args


def main():
  try:
    start = time.time()
    args = parse_args()
    
    log = log_init()
    
    finding_names = args.find_person.split()
    print('People we need to identify:', finding_names)
    print('Used device: ', config.DEVICE)

    log = write_log(old_log=log,
                    new_message= "People we need to identify: {}".format(finding_names),
                    type="string + enter")

    log = write_log(old_log=log,
                    new_message= "Used device: {}".format(config.DEVICE),
                    type="string + enter")
    
    input_paths, input_names = return_paths(args.original_dataset_path, 'input')
    anchor_paths, anchor_labels = return_paths(args.anchor_dataset_path, 'anchor')

    people_in_anchor = np.unique(anchor_labels).tolist()
    check_if_exist = [person in people_in_anchor for person in finding_names]

    if False in check_if_exist:
      raise Exception("Don't have the people you're finding in your anchor dataset")

    mtcnn, infer_model = create_facenet_models()
    FIQA_net = FIQA_network()
    print('-----Initializing FIQA and FaceNet models-----')

    if FIQA_net and mtcnn and infer_model is not None:
        print('-----Done initialized FIQA and FaceNet models-----')

        log = write_log(old_log=log,
                        new_message="-----Done initialized FIQA and FaceNet models-----",
                        type="string + enter")
    else:
        raise Exception('-----Failed to create FIQA + FaceNet models-----')

    append_df = []
    for batch_index in range(len(input_paths)):
      df, input_img = face_detection(input_paths[batch_index], input_names[batch_index], anchor_paths, anchor_labels, mtcnn, infer_model, finding_names)

      log = write_log(old_log = log,
                      new_message = 'Dataframe after face detection',
                      type = "string + enter")

      log = write_log(old_log = log,
                      new_message = df,
                      type = "dataframe + enter")

      if input_img:
        torch.cuda.empty_cache()
        df, input_img = FIQA(df, input_img, fiqa_net)
        df, input_img = get_smile_score(df, input_img)
        append_df.append(df)

        log = write_log(old_log = log,
                        new_message = 'Dataframe after FIQA + smile score',
                        type = "string + enter")

        log = write_log(old_log = log,
                        new_message = df,
                        type = "dataframe + enter")

      log = write_log(old_log = log,
                      new_message = "-----Finished batch {} -----".format(batch_index + 1),
                      type = "string + enter")

      end = time.time()

      log = write_log(old_log = log,
                      new_message = "-----Time since start for batch {batch_index}: {time:.4g} -----".format(batch_index = batch_index + 1, time = end-start),
                      type = "string + enter")

      print('-----Finished batch {} -----'.format(batch_index + 1))
      print('Time since start: {0:.4g}'.format(end-start))

    log = write_log(old_log=log,
                    new_message="Processing DataFrame",
                    type="string + enter")

    if not append_df:
      raise Exception("Can't find any images")
    df_final = pd.concat(append_df)

    df_final.sort_values(by = 'smile score average', ascending = False, inplace = True)
    df_final.reset_index(drop = True)
    
    finding_ids = [finding_names[x:x+1] for x in range(0, len(finding_names), 1)]
    if args.find_all and len(finding_ids) > 1:
        df_final = df_final[df_final['ids'].apply(lambda x: check_ids_equal(x, finding_ids))]
        df_final.reset_index(drop = True)

    log = write_log(old_log=log, 
                    new_message=df_final, 
                    type="dataframe + enter")

    df_final = df_final.iloc[:args.number_of_images]
    input_img, input_img_resized, input_shape_flag = read_images(list(df_final['paths']), purpose = 'input')

    print('-----Creating video-----')

    log = write_log(old_log=log,
                    new_message="-----Creating video-----",
                    type="string + enter")
    
    log = write_log(old_log=log,
                    new_message="Visualize bounding boxes: {}".format(args.visualize_boxes),
                    type="string + enter")
    
    if args.visualize_boxes:
      if not input_shape_flag:
        bboxes = df_final['bboxes'].tolist()
        df_final['bboxes'] = rescale_bboxes(bboxes, input_img, input_img_resized)

        log = write_log(old_log = log,
                        new_message = "New dataframe with rescaled bounding boxes (for visualization purpose)",
                        type = "string + enter")

        log = write_log(old_log = log,
                        new_message = df_final,
                        type = "dataframe + enter")

      input_img = visualizing_bounding_boxes(df_final, input_img_resized)
      del input_img_resized

    make_video(img_list = input_img,
               output_path = args.output_path,
               effect_speed = args.effect_speed,
               duration = args.duration,
               fps = args.fps,
               fraction = args.fraction)

    end = time.time()

    print('-----Done creating video-----')

    log = write_log(old_log=log,
                    new_message="-----Done creating video-----",
                    type="string + enter")

    print('-----DONE-----. Total time: ', end-start)

    log = write_log(old_log=log,
                    new_message= "DONE. Total time: {0:.4g} ".format(end-start),
                    type="string + enter")

  except Exception as e:
    print(e)

  finally:
    if args.log:
        log_final(log)


if __name__ == '__main__':
    main()
