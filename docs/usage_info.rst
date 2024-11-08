==========
Usage Info
==========

This project can be used as a python module or as an standalone application. The
**Encoder** and **Decoder** are intended to be used in together, but it is
possible to use them with other modules see, for more information about the
usage of each of the modules see :ref:`API section <vocoder_api-label>`

Encoder Usage
==============
The encoder is in charge of analysing the speech signal and will generate the
LPC coefficients, gain and pitch of the signal.

Encoder Python Module
---------------------
To use it as a python module just import *LpcEncoder* class and create an
instance of it.

.. code-block:: python

    >>> from lpc_vocoder.encode import LpcEncoder
    >>> encoder = LpcEncoder()
    >>>
    >>> # Load an audio file
    >>> encoder.load_file(path_to_file, window_size=480)
    >>>
    >>> # Process the file
    >>> encoder.encode_signal()
    >>>
    >>> # Save encoded data to a binary file
    >>> encoder.save_data(path_to_output_binary)



Encoder Standalone Application
------------------------------
To use it as an standalone application just call *lpc_encoder*, this application
requires the following arguments:

#. audio_file: this is the input file that will be processed
#. encoded_file: output file with the encoded data

There are other optional parameters that can be found from the help menu:

.. code-block:: console

    (venv) bash-5.1$ lpc_encoder -h
    usage: lpc_encoder [-h] [--order ORDER] [--frame_size FRAME_SIZE] [--overlap OVERLAP] [-d] audio_file encoded_file

    Encode a .wav signal using LPC

    positional arguments:
      audio_file            Name of the input file
      encoded_file          Name of the output file

    options:
      -h, --help            show this help message and exit
      --order ORDER         Order of the LPC filter, default '10'
      --frame_size FRAME_SIZE
                            Frame Size, if not provided it will use a 30ms window based on the sample rate
      --overlap OVERLAP     Overlap as percentage (0-100), default '50'
      -d, --debug           Enable debug messages


The following code snippet has an usage example:

.. code-block:: console

    (venv) bash-5.1$ lpc_encoder lpc_vocoder/test/audios/the_boys.flac the_boys --order=40 --frame_size=512
    2024-11-06 17:02:53,815 [INFO] Encoding file 'lpc_vocoder/test/audios/the_boys.flac'
    (venv) bash-5.1$
    (venv) bash-5.1$ ls the_boys*
    the_boys.bin


Decoder Usage
=============
The decoder is the companion module from the *Encoder*, it will receive a binary
file with the encoded data and process it to recover the original signal.

Decoder Python Module
---------------------
To use it as a python module just import *LpcDecoder* class and create an
instance of it.

.. code-block:: python

    >>> from lpc_vocoder.decode import LpcDecoder
    >>> decoder = LpcDecoder()
    >>>
    >>> # Load an audio binary file
    >>> decoder.load_data_file(path_to_file)
    >>>
    >>> # Process the file
    >>> decoder.decode_signal()
    >>>
    >>> # Save encoded data to a wav file
    >>> decoder.save_audio(Path("audio.wav"))
    >>>
    >>> # Play the decoded signal
    >>> decoder.play_signal()


Decoder Standalone Application
------------------------------
To use it as an standalone application just call *lpc_decoder*, this application
requires the following arguments:

#. encoded_file: binary file produced by the encoder
#. audio_file: audio file with the reconstructed signal

There are other optional parameters that can be found from the help menu:

.. code-block:: console

    (venv) bash-5.1$ lpc_decoder -h
    usage: lpc_decoder [-h] [--play] [-d] encoded_file audio_file

    Encode a .wav signal using LPC

    positional arguments:
      encoded_file  Name of the input file
      audio_file    Name of the output file

    options:
      -h, --help    show this help message and exit
      --play        Play the decoded signal
      -d, --debug   Enable debug messages


The following code snippet has an usage example:

.. code-block:: console

    (venv) bash-5.1$ lpc_decoder the_boys.bin the_boys.wav
    2024-11-06 17:04:17,933 [INFO] Decoding file 'the_boys.wav'
    (venv) bash-5.1$
    (venv) bash-5.1$ ls the_boys*
    the_boys.bin  the_boys.wav
