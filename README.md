# mal-anime-recommender

## Steps to get this to work locally:
1. `python3 -r requirements.txt`
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
  
6. Retrieving anime details:
   - requires anime ids
   - `python3 mal_api.py anime_details param1`
   - append_status = "param1": boolean (True/False) to determine if we want to append to an existing anime_ids file
   - follow instructions
  
8. Retrieving videos:
   - requires video urls from anime details step
   - `python3 video_parser.py get_videos`
  
9. Trimming available videos:
   - requires videos
   - `python3 mal_api.py trim_all param1 param2`
   - start_percentage = "param1": to what percent to clip off
   - end_percentage = "param2": from what percent to clip off
   - if you only want 20-80% -> start_percentage, end_percentage = 20, 80
  
10. Generate random frames:
    - requires videos
    - `python3 mal_api.py random_frames_all params1`
    - frames = "params1": the number of frames to save per video
   
11. Removing violating frames
    - requires frames
    - `python3 mal_api.py remove_bad_frames`
    - function is really slow and does not remove every violating frame
    - highly recommended to go through yourself
