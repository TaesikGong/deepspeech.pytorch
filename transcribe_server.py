import argparse
import warnings
import socket
import sys, time

warnings.simplefilter('ignore')

from decoder import GreedyDecoder

import torch

from data.data_loader import SpectrogramParser
from model import DeepSpeech
import os.path
import json

parser = argparse.ArgumentParser(description='DeepSpeech transcription')
# parser.add_argument('--model-path', default='models/deepspeech_final.pth', help='Path to model file created by training')
# parser.add_argument('--audio-path', default='audio.wav',help='Audio file to predict on')

# parser.add_argument('--model-path', default='models/librispeech_pretrained.pth', help='Path to model file created by training')
parser.add_argument('--model-path', default='models/deepspeech_final.pth', help='Path to model file created by training')
# parser.add_argument('--model-path', default='models/libri_finetune_final.pth', help='Path to model file created by training')
# parser.add_argument('--audio-path', default='audios/cat1.wav',help='Audio file to predict on')
parser.add_argument('--audio-path', default='audios/cat/00f0204f_nohash_1.wav',help='Audio file to predict on')
# parser.add_argument('--audio-path', default='audios/cat/0a196374_nohash_0.wav',help='Audio file to predict on')

parser.add_argument('--cuda', action="store_true", help='Use cuda to test model')
parser.add_argument('--decoder', default="greedy", choices=["greedy", "beam"], type=str, help="Decoder to use")
parser.add_argument('--offsets', dest='offsets', action='store_true', help='Returns time offset information')
beam_args = parser.add_argument_group("Beam Decode Options", "Configurations options for the CTC Beam Search decoder")
beam_args.add_argument('--top-paths', default=1, type=int, help='number of beams to return')
beam_args.add_argument('--beam-width', default=10, type=int, help='Beam width to use')
beam_args.add_argument('--lm-path', default=None, type=str,
                       help='Path to an (optional) kenlm language model for use with beam search (req\'d with trie)')
beam_args.add_argument('--alpha', default=0.8, type=float, help='Language model weight')
beam_args.add_argument('--beta', default=1, type=float, help='Language model word bonus (all words)')
beam_args.add_argument('--cutoff-top-n', default=40, type=int,
                       help='Cutoff number in pruning, only top cutoff_top_n characters with highest probs in '
                            'vocabulary will be used in beam search, default 40.')
beam_args.add_argument('--cutoff-prob', default=1.0, type=float,
                       help='Cutoff probability in pruning,default 1.0, no pruning.')
beam_args.add_argument('--lm-workers', default=1, type=int, help='Number of LM processes to use')


args = parser.parse_args()


def decode_results(model, decoded_output, decoded_offsets):
    results = {
        "output": [],
        "_meta": {
            "acoustic_model": {
                "name": os.path.basename(args.model_path)
            },
            "language_model": {
                "name": os.path.basename(args.lm_path) if args.lm_path else None,
            },
            "decoder": {
                "lm": args.lm_path is not None,
                "alpha": args.alpha if args.lm_path is not None else None,
                "beta": args.beta if args.lm_path is not None else None,
                "type": args.decoder,
            }
        }
    }
    results['_meta']['acoustic_model'].update(DeepSpeech.get_meta(model))

    for b in range(len(decoded_output)):
        for pi in range(min(args.top_paths, len(decoded_output[b]))):
            result = {'transcription': decoded_output[b][pi]}
            if args.offsets:
                result['offsets'] = decoded_offsets[b][pi].tolist()
            results['output'].append(result)
    return results


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    model = DeepSpeech.load_model(args.model_path, cuda=args.cuda)
    model.eval()

    labels = DeepSpeech.get_labels(model)
    audio_conf = DeepSpeech.get_audio_conf(model)

    if args.decoder == "beam":
        from decoder import BeamCTCDecoder

        decoder = BeamCTCDecoder(labels, lm_path=args.lm_path, alpha=args.alpha, beta=args.beta,
                                 cutoff_top_n=args.cutoff_top_n, cutoff_prob=args.cutoff_prob,
                                 beam_width=args.beam_width, num_processes=args.lm_workers)
    else:
        decoder = GreedyDecoder(labels, blank_index=labels.index('_'))

    parser = SpectrogramParser(audio_conf, normalize=True)



    ###--- server setup
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = ('', 10000)
    print('starting up on %s port %s' % server_address)
    sock.bind(server_address)
    sock.listen(1)
    file_name = 'data/recorded.wav'
    bFileFound = 0
    while True:
        # Wait for a connection
        connection, client_address = sock.accept()
        print('connection from', client_address)
        try:
            # while True:
            #     recv_data = connection.recv(1024)
            #     while recv_data != b'ok':
            #         recv_data = connection.recv(1024)
            #         print('ok waiting')
            #     print('ok received')
            recv_data = connection.recv(1024)
            recv_file = open(file_name, 'wb')
            while recv_data:
                recv_file.write(recv_data)
                recv_data = connection.recv(1024)

            recv_file.close()
            print('download complete')

            start = time.time()
            #inference
            spect = parser.parse_audio(file_name).contiguous()
            parsing_time = time.time() - start

            spect = spect.view(1, 1, spect.size(0), spect.size(1))
            out = model(spect)
            inferring_time = time.time() - parsing_time - start

            decoded_output, decoded_offsets = decoder.decode(out.data)
            decoding_time = time.time() - inferring_time - start

            print('time for parsing: %0.4f,\t inferring: %0.4f,\t decoding: %0.4f'%(parsing_time,inferring_time,decoding_time))
            print(json.dumps(decode_results(model, decoded_output, decoded_offsets)))
        except ConnectionResetError:#connection is broken by a client
            pass
        finally:
            # Clean up the connection
            print('connection close')
            connection.close()




