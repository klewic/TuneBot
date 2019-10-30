# TuneBot

The TuneBot is a set of generative adversarial networks designed with the
intention of randomly generating music in the form of midi files.

This is a very involved project that I've been chipping away at for over a year
now.  I've had to find a way to build a data set from scratch using an archive
of midi files from Geocities websites (rip) on top of tweaking the design of the
neural networks endlessly hoping for output that isn't a scrambled bunch of
nonsense.

I've only included a small portion of the data in the Generate_Data directory.
However, the tools to convert the midi files to text are all present within that
directory.  Feel free to make your own data set using your own midi collection.

This project has gone through many iterations, and the scrambled nature of all
of my code files shows it.  This project is also far from over.  At the moment,
the TuneBot NoteGenerator seems to ignore the random input provided to it and
outputs mostly the same song every time.  However, I've generated a variety of
models that each produce a distinct, relatively nice sounding tune (I've
included four of the generated songs in the root of the project - give 'em a
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
