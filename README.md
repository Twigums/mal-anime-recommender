# mal-anime-recommender

## Steps to get this to work locally:
1. `pip3 install -r requirements.txt`
2. Get an api (I will not share my api token... :( )
   - run `python3 mal_api.py get_api`
   - follow instructions
  
3. Now that you have a client id and access token,
   - refreshing api: `python3 mal_api.py refresh_api`
   - you can choose what to do now
  
4. Retrieving anime ids:
   - `python3 mal_api.py anime_ids param1 param2 param3`
   - limit = "param1": how many top animes you want
   - init_offset = "param2": if you want to offset the ranking (ex: start at anime #50 -> init_offset = 50)
   - append_status = "param3": boolean (True/False) to determine if we want to append to an existing anime_ids file
  
5. Retrieving anime details:
   - requires anime ids
   - `python3 mal_api.py anime_details param1`
   - append_status = "param1": boolean (True/False) to determine if we want to append to an existing anime_ids file
   - follow instructions
  
  
6. Retrieving user information:
   - requires api
   - `python3 mal_api.py user_info param1 param2`
   - username = "param1": username to retrieve info from
   - append_status = "param2": boolean (True/False) to determine if we want to append to an existing user info file(s)
  
7. Retrieving videos:
   - requires video urls from anime details step
   - `python3 video_parser.py get_videos`

8. Trimming available videos:
   - requires videos
   - `python3 mal_api.py trim_all param1 param2`
   - start_percentage = "param1": to what percent to clip off
   - end_percentage = "param2": from what percent to clip off
   - if you only want 20-80% -> start_percentage, end_percentage = 20, 80
  
9. Generate random frames:
    - requires videos
    - `python3 mal_api.py random_frames_all params1`
    - frames = "params1": the number of frames to save per video
   
10. Removing violating frames
    - requires frames
    - `python3 mal_api.py remove_bad_frames`
    - function is really slow and does not remove every violating frame
    - highly recommended to go through yourself
   
11. Making ImageFolder
    - requires frames and classes (if move user info, 2nd line)
    - `python3 image_to_tensor.py move_to_classes param1 param2`
    - `python3 image_to_tensor.py user_move_to_classes param1 param2 param3 param4`
    - path_to_make = "param1": destination of where to make the ImageFolder
    - path_to_folder = "param2": where the current folder with frames is
    - path_user_labels = "param3": where the text file for user scores is
    - path_user_animes = "param4": where the text file for user anime is
   
12. (Optional) Make noisy images
    - requires frames
    - `python3 image_to_tensor.py add_noise param1 param2 param3`
    - path_to_images = "param1": where the current folder with frames is
    - path_to_noised_images = "param2": destination of where to make the new frames
    - std = "param3": standard deviation of the normal Gaussian distribution to specify
    - to replicate my set, std was 0.8 and 3.
   
13. Running general model
    - requires ImageFolder dataset and CUDA
    - `python3 CNN.py`
    - remember to specify filepaths
    - path_to_data is where you have the training dataset
    - path_to_test is where you have the testing dataset
    - path_to_model is where you save/load the model
    - additional params: need_save and noRound can be specified (default True/False respectively to save model and round anime scores)
   
14. Running user model
    - requires ImageFolder dataset, CUDA, saved model
    - `python3 test_model.py`
    - remember to specify filepaths
    - path_to_user_data is where you have the user dataset
    - path_to_recommend_data is where you have the small ImageFolder dataset
    - path_to_model is where you saved the general model
    - path_to_save_model is where you want to save/load the user model
    - additional params: need_save and noRound can be specified (default `True`/`False` respectively to save model and round anime scores)
    - if you want to have the model recommend, specify needs_save as `False`
