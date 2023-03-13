import torch
from data import LibriSpeechDataset, get_dataloader
from model import CausalConformerModel
from tokenizer import SentencePieceTokenizer
from torchaudio.functional import rnnt_loss
from torchmetrics.functional import char_error_rate, word_error_rate
from tqdm import tqdm


def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "./librispeech_small/artifacts/36b6c18c29004b50890ce3116092c6b1/artifacts/model_50.pth"
    with open(model_path, "rb") as f:
        cpt = torch.load(f)
    model_state = cpt["model"]
    model_args = cpt["model_args"]
    model = CausalConformerModel(**model_args).to(DEVICE)
    model.load_state_dict(model_state)
    tokenizer = SentencePieceTokenizer(model_file_path="./vocabs/librispeech_1024_bpe.model")
    dataset = LibriSpeechDataset(
        resampling_rate=16000,
        tokenizer=tokenizer,
        json_file_path="./json/librispeech_dev-other.json",
    )
    dataloader = get_dataloader(
        dataset,
        batch_sec=60,
        num_workers=1,
        pad_idx=tokenizer.pad_token_id,
        pin_memory=True,
    )
    dev_cer = 0
    dev_wer = 0
    counter = 0
    bar = tqdm(total=len(dataset))
    with torch.no_grad():
        model.eval()
        for _, benc_input, bpred_input, benc_input_length, bpred_input_length, baudio_sec in dataloader:
            benc_input = benc_input.to(DEVICE)
            bpred_input = bpred_input.to(DEVICE)
            bpadded_output, bpadded_ctc_log_probs, bsubsampled_enc_input_length = model(
                padded_enc_input=benc_input,
                enc_input_lengths=benc_input_length,
                padded_pred_input=bpred_input,
                pred_input_lengths=bpred_input_length,
            )
            """
            loss = rnnt_loss(
                logits=bpadded_output,
                targets=bpred_input,
                logit_lengths=bsubsampled_enc_input_length.to(DEVICE),
                target_lengths=bpred_input_length.to(DEVICE),
                blank=tokenizer.blank_token_id,
                reduction="sum",
            )
            """
            bhyp_token_indices = model.streaming_greedy_inference(
                enc_inputs=benc_input, enc_input_lengths=benc_input_length
            )
            bhyp_text = tokenizer.batch_token_ids_to_text(bhyp_token_indices)
            bans_token_indices = [
                bpred_input[i, : bpred_input_length[i]].tolist() for i in range(bpred_input.shape[0])
            ]
            bhyp_text = tokenizer.batch_token_ids_to_text(bhyp_token_indices)
            bans_text = tokenizer.batch_token_ids_to_text(bans_token_indices)

            dev_cer += char_error_rate(bhyp_text, bans_text) * benc_input.shape[0]
            dev_wer += word_error_rate(bhyp_text, bans_text) * benc_input.shape[0]
            counter += benc_input.shape[0]

            bar.update(benc_input.shape[0])
            bar.set_postfix({"WER": dev_wer / counter, "CER": dev_cer / counter})

    print(dev_cer / counter)
    print(dev_wer / counter)

    with open("streaming_result.txt", "w") as f:
        f.write("CER: {}\n".format(dev_cer / counter))
        f.write("WER: {}\n".format(dev_wer / counter))


if __name__ == "__main__":
    main()
