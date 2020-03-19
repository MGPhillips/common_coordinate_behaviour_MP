# Filename and path to behavioral video for analysis
videofolder = '../videos/'
videotype='.avi' #type of videos to analyze

#########################################################################################
# Analysis Network parameters
#########################################################################################

# These variables should be changed so that the right networks is loaded for analysis
# (Typicaly just copy them over from myconfig.py)
scorer = 'Mackenzie'
Task = 'reaching'
date = 'Jan30'
trainingsFraction = 0.95
resnet = 50
snapshotindex = -1
shuffle = 1

storedata_as_csv=False #if true then the time series of poses will (also) be saved as csv.

# Note the data is always saved in hdf - format which is an efficient format
# that easily allows to load the full pandas multiarray at a later stage

#########################################################################################
## For plotting (MakingLabeledVideo.py / MakingLabeledVideo_fast.py)
#########################################################################################

trainingsiterations = 500  # type the number listed in the h5 file containing the pose estimation data. The video will be generated
#based on the labels for this network state.

pcutoff = 0.1  # likelihood cutoff for body part in image

# delete individual (labeled) frames after making video? (note there could be many...)
deleteindividualframes = False
alphavalue=.6 # "strength/transparency level of makers" in individual frames (Vary from 0 to 1. / not working in "MakingLabeledVideo_fast.py")
dotsize = 7
colormap='hsv' #other colorschemes: 'cool' and see https://matplotlib.org/examples/color/colormaps_reference.html
