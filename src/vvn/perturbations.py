import random


"""
A set of perturbations for video classification + video prediction models.
We consider perturbations across the x,y,z dimensions.

1. Perturbation of a single pixel on a single frame
2. Perturbation of a single pixel on all frames
3. Perturbation of all pixels on a single frame
4. Perturbation of all pixels on all frames
5. Black-out of a single randomly selected frame
6. Swapping of 2 random frames

"""

# set the random seed
# random.seed(10)

def perturb_sp_sf(video, frameNum, pixelW, pixelH, epsilon):
    video[frameNum][pixelW][pixelH] += epsilon
    return video
        
def perturb_sp_af(video, pixelW, pixelH, epsilon):
    for frame in video:
        frame[pixelW][pixelH] += epsilon

    return video

def perturb_ap_sf(video, frameNum, epsilon):
    for i in range(len(video[frameNum])):
        for j in range(len(video[frameNum][i])):
            video[frameNum][i][j] += epsilon

    return video

def perturb_ap_af(video, epsilon):
    for frameNum in range(len(video)):
        for i in range(len(video[frameNum])):
            for j in range(len(video[frameNum][i])):
                video[frameNum][i][j] += epsilon

    return video
    
def black_out(video, frameNum):
    for i in range(len(video[frameNum])):
        for j in range(len(video[frameNum][i])):
            # set to black
            video[frameNum][i][j] = 0

    return video

def swap_frames(video, frame1, frame2):
    tmp = video[frame1]
    video[frame1] = video[frame2]
    video[frame2] = tmp
    return video


if __name__=="__main__":
    pass
