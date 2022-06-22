def rm_ext(filename):
    filename=os.path.basename(filename)
    newpath =os.path.splitext(filename)[0]
    
    if "." in newpath:
        newpath=rm_ext(newpath)
    return(newpath)  