from pydub import AudioSegment
from pydub.utils import make_chunks
#import sys

def split(wav_file,out_name,timestep=1000):
    myaudio = AudioSegment.from_file(wav_file , "wav")
    chunk_length_ms = timestep # pydub calculates in millisec
    #import pdb;pdb.set_trace()
    chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec
    if len(chunks) == 41:
        chunks.pop()
    elif len(chunks)<40:
        return
    

    #Export all of the individual chunks as wav files
    for i, chunk in enumerate(chunks):
        # if i == len(chunks)-1:continue
        chunk_name = "%s-%04d.wav"%(out_name,i)
        print ("exporting", chunk_name)
        chunk.export(chunk_name, format="wav")
