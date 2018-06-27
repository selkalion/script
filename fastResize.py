import os
import bulk_resize as br                                                                                                    

subdirs = [x[0] for x in os.walk(".\\cropped")]
subdirs.remove(".\\cropped")                                                                            
for subdir in subdirs:
    out = subdir.replace(".\\cropped", ".\\resized_for_training", 1)                                                                                                         
    #print(subdir)
    if not os.path.exists(out):
        print(out)
        br.bulk_resize(subdir, out) 
        