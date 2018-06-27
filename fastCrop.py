import os
import bulk_convert as bc                                                                                                    

subdirs = [x[0] for x in os.walk(".\\raw")]
subdirs.remove(".\\raw")                                                                            
for subdir in subdirs:
    out = subdir.replace(".\\raw", ".\\cropped", 1)                                                                                                         
    #print(subdir)
    if not os.path.exists(out):
        print(out)
        bc.bulk_convert(subdir, out) 
        