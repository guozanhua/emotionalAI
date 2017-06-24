"""
After moving all the files using the 1_ file, we run this one to extract
the images from the videos and also create a data file we can use
for training and testing later.
"""
import csv
import glob
import os, sys
import os.path
from subprocess import call

def extract_files():
    """After we have all of our videos split between train and test, and
    all nested within folders representing their classes, we need to
    make a data file that we can reference when training our RNN(s).
    This will let us keep track of image sequences and other parts
    of the training process.

    We'll first need to extract images from each of the videos. We'll
    need to record the following data in the file:

    [train|test], class, filename, nb frames

    Extracting can be done with ffmpeg:
    `ffmpeg -i video.mpg image-%04d.jpg`
    """
    data_file = []
    folders = ['./train/', './test/']
    seq_len = 40
    def rescale_list(input_list, size):
        """Given a list and a size, return a rescaled/samples list. For example,
        if we want a list of size 5 and we have a list of size 25, return a new
        list of size five which is every 5th element of the origina list."""
        #import pdb;pdb.set_trace()
        try:
            assert len(input_list) >= size
        except:
            return []

        # Get the number to skip between iterations.
        skip = len(input_list) // size

        # Build our new output.
        output = [input_list[i] for i in range(0, len(input_list), skip)]

        # Cut off the last one if needed.
        return output[:size]

    f = open('failedfiles.csv','w')
    for folder in folders:
        class_folders = glob.glob(folder + 'vids/%s'%sys.argv[1])
        #class_folders = glob.glob(folder + 'vids/ApplyEyeMakeup')
        #class_folders = glob.glob(folder + 'vids/test')

        for vid_class in class_folders:
            class_files = glob.glob(vid_class + '/*.mp4')
            #class_files = glob.glob('./train/vids/ApplyEyeMakeup/v_ApplyEyeMakeup_g09_c01.avi')
            #class_files = glob.glob('./train/vids/test/2013DisneysTwilightZoneTowerofTerror-.mp4')
            for video_path in class_files:
                # Get the parts of the file.
                #import pdb; pdb.set_trace()
                video_parts = get_video_parts(video_path)

                train_or_test, srctype, classname, filename_no_ext, filename = video_parts
                #import pdb;pdb.set_trace()
                # Check if this class exists.
                for t in ['imgs','wavs','wavsfull','specs','imgsmerge','imgsrescale']:
                    if not os.path.exists(train_or_test + '/'+t+'/' + classname):
                        print("Creating folder for %s/%s" % (train_or_test, classname))
                        os.makedirs(train_or_test + '/'+t+'/' + classname)
                # Only extract if we haven't done it yet. Otherwise, just get
                # the info.
                if not check_already_extracted(video_parts):
                    # Now extract it.
                    src = train_or_test + '/vids/' + classname + '/' + \
                        filename
                    dest = train_or_test + '/imgs/' + classname + '/' + \
                        filename_no_ext + '-%04d.jpg'
                    dest1 = train_or_test + '/imgsrescale/' + classname + '/' + \
                        filename_no_ext
                    dest2 = train_or_test + '/wavsfull/' + classname + '/' + \
                        filename_no_ext + '.wav'
                    dest3 = train_or_test + '/wavs/' + classname + '/' + \
                           filename_no_ext #+ '-%04d.wav'
                    dest4 = train_or_test + '/specs/' + classname + '/'
                    dest5 = train_or_test + '/imgsrescale/' +classname + '/'
                    dest6 = train_or_test + '/imgsmerge/'  +classname + '/'
                    from ffprobe3 import FFProbe
                    metadata = FFProbe(src)
                    if len(metadata.streams) != 2: continue
                    try:
                        duration = metadata.audio[0].duration_seconds()
                    except:
                        continue
                    timestep = float(duration)/seq_len
                    print (duration,timestep)
                    os.system("ffmpeg -i %s %s" %(src, dest))
                    #import pdb; pdb.set_trace()
                    os.system("ffmpeg -i %s %s" % (src, dest2))
                    imgs = glob.glob(train_or_test + '/imgs/' + classname + '/' + \
                        filename_no_ext+'*.jpg')
                    rescale_imgs = rescale_list(sorted(imgs),seq_len)
                    if rescale_imgs == []:
                        f.write("glob,%s\n"%filename_no_ext)
                        continue
                    for i,im in enumerate(rescale_imgs):
                        os.system("cp %s %s-%04d.jpg"%(im,dest1,i))
                    # from ffprobe3 import FFProbe
                    # metadata = FFProbe(src)
                    # duration = metadata.streams[0].duration_seconds()
                    # timestep = float(duration)/seq_len
                    # print (duration,timestep)
                    import split_audio as spa
                    spa.split(dest2,dest3,timestep=timestep*1000)
                    import audio2spec_scipy2 as a2s
                    import img_concat as imcat
                    #from glob import glob
                    names = glob.glob(dest3+'*.wav')
                    if len(names) == 0:
                        continue
                    try:
                        assert len(rescale_imgs)==len(names)
                    except:
                        f.write("audio,%s\n"%filename_no_ext)
                        continue
                         #for t in [('imgs','.jpg'),('wavs','.wav'),('wavsfull','.wav'),('specs','.png'),('imgsmerge','.jpg'),('imgsrescale','.jpg')]:
                         #    if os.path.exists(train_or_test + '/'+t[0]+'/' + classname+'/'+filename_no_ext+'*'):
                         #       os.system("rm -f %s"%(train_or_test + '/'+t[0]+'/' + classname+'/'+filename_no_ext+'*'))
                            #os.remove(train_or_test + '/'+t[0]+'/' + classname+'/'+filename_no_ext+t[1])
                    for wf in names:
                        #a2s.graph_spectrogram(wf,dest4+os.path.basename(wf).split('.')[0])
                        a2s.plotstft(wf, plotpath=dest4+os.path.basename(wf).split('.')[0])
                        imcat.concat(dest5+os.path.basename(wf).split('.')[0]+'.jpg',\
                                     dest4+os.path.basename(wf).split('.')[0]+'.png',\
                                     dest6+os.path.basename(wf).split('.')[0])

                    #os.system("ffmpeg -t 60 -acodec copy -i %s %s" % (src, dest3))
                    #call(["ffmpeg", "-r", '1', "-i", src, dest])
                    #call(["ffmpeg", "-vn", "-acodec", "pcm_s16le", "-ar", '44100', "-ac", '2', "-i", src, dest2])
                    #call(["ffmpeg", "-t", '60', "-acodec", "copy", "-i", dest2, dest3])

                # Now get how many frames it is.
                nb_frames = get_nb_frames_for_video(video_parts)

                data_file.append([train_or_test, classname, filename_no_ext, nb_frames])

                print("Generated %d frames for %s" % (nb_frames, filename_no_ext))
    f.close()
    with open('data_file.csv', 'w') as fout:
        writer = csv.writer(fout)
        writer.writerows(data_file)

    print("Extracted and wrote %d video files." % (len(data_file)))

def get_nb_frames_for_video(video_parts):
    """Given video parts of an (assumed) already extracted video, return
    the number of frames that were extracted."""
    train_or_test, srctype,classname, filename_no_ext, _ = video_parts
    generated_files = glob.glob(train_or_test + '/imgsmerge/' + classname + '/' +
                                filename_no_ext + '*.jpg')
    return len(generated_files)

def get_video_parts(video_path):
    """Given a full path to a video, return its parts."""
    parts = video_path.split('/')
    filename = parts[4]
    filename_no_ext = filename.split('.')[0]
    classname = parts[3]
    srctype = parts[2]
    train_or_test = parts[1]

    return train_or_test, srctype, classname, filename_no_ext, filename

def check_already_extracted(video_parts):
    """Check to see if we created the -0001 frame of this file."""
    train_or_test, srctype,classname, filename_no_ext, _ = video_parts
    '''import shutil
    shutil.rmtree(train_or_test + '/imgs/' )
    shutil.rmtree(train_or_test + '/imgsrescale/' )
    shutil.rmtree(train_or_test + '/imgsmerge/' )
    shutil.rmtree(train_or_test + '/wavs/' )
    shutil.rmtree(train_or_test + '/wavsfull/' )
    shutil.rmtree(train_or_test + '/specs/' )
    os.makedirs(train_or_test + '/imgs/')
    os.makedirs(train_or_test + '/imgsrescale/')
    os.makedirs(train_or_test + '/imgsmerge/')
    os.makedirs(train_or_test + '/wavs/')
    os.makedirs(train_or_test + '/wavsfull/')
    os.makedirs(train_or_test + '/specs/')'''
    return bool(os.path.exists(train_or_test + '/imgs/' + classname +
                               '/' + filename_no_ext + '-0001.jpg'))

def main():
    """
    Extract images from videos and build a new file that we
    can use as our data input file. It can have format:

    [train|test], class, filename, nb frames
    """
    extract_files()

if __name__ == '__main__':
    main()
