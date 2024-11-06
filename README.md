# Vocoder LPC for speech signals

This project implements a Linear Predictive Coding (LPC) vocoder, which includes
**encoder** and a **decoder** modules. The LPC vocoder uses standard LPC 
calculation based on the [Wiener Filter](https://en.wikipedia.org/wiki/Wiener_filter)
and the [source-filter model](https://en.wikipedia.org/wiki/Source%E2%80%93filter_model)

## Encoder Usage
The encoder can be imported as module:
```
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
```

Or it can be executed as a standalone application:
```
(venv) bash-5.1$ 
(venv) bash-5.1$ lpc_encoder lpc_vocoder/test/audios/the_boys.flac the_boys --order=40 --frame_size=512
2024-11-06 17:02:53,815 [INFO] Encoding file 'lpc_vocoder/test/audios/the_boys.flac'
(venv) bash-5.1$ 
(venv) bash-5.1$ ls the_boys*
the_boys.bin
```

## Decoder Usage
Similar to the encoder, the decoder can be used as a module:
```
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
```
Or as a standalone application:
```
(venv) bash-5.1$ 
(venv) bash-5.1$ lpc_decoder the_boys.bin the_boys.wav
2024-11-06 17:04:17,933 [INFO] Decoding file 'the_boys.wav'
(venv) bash-5.1$ 
(venv) bash-5.1$ ls the_boys*
the_boys.bin  the_boys.wav
```
