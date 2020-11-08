# TuneBot

The TuneBot is a set of generative adversarial networks designed with the
intention of randomly generating music in the form of midi files.

This is a very involved project that I've been chipping away at for over a year
now.  From figuring out a way to build a data set from scratch using an archive
of midi files from old Geocities websites to tweaking the design of the
neural networks endlessly hoping for output that isn't a scrambled bunch of
nonsense.

I only included a small portion of the data I used in the Generate_Data directory.
However, the tools to convert any midi file to text are all present within that
directory.  Feel free to make your own data set using your own midi collection 
and see what happens.

This project has gone through many iterations and will go through more when I find
the time.  At the moment, the TuneBot NoteGenerator ignores the random input provided
thus models tend to converge to a single song.  However, I've generated a variety of
models that each produce a distinct, relatively nice sounding tune (I've
included four of the generated songs in the root dir of this project - give 'em a
listen!).  I have some ideas for improving upon the GAN's that will likely come
with a rewrite of everything in Pytorch (Tensorflow just wasn't cutting it for
an experimental project like this).

To give a rough overview of the struction of this project:
The GAN's used to build the models for generating meter and notes can be found
in the Generate_Meter and Generate_Notes directories.  You can play a tune via
the PlayClip.py file in the root directory which calls upon MeterGenerator.py
and NoteGenerator.py.

I'll cut the explanation short there.  This was primarily a fun learning
experience for myself, but feel free to contact me if you want to ask any deeper
questions regarding my methods.
