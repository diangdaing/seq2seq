import torch
from model import Encoder, Decoder, Seq2Seq
from utils import load_dataset

MAX_LEN = 10

def translate(model, sentence, DE, EN):
    model.eval()
    with torch.no_grad():
        # Tokenize input sentence and convert to tensor
        # 标记输入句子并转换为张量
        tokens = [t.lower() for t in sentence.split()]
        src_indexes = [DE.vocab.stoi[t] for t in tokens]
        src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)

        # Pass input through encoder to get context vector
        # 通过编码器传递输入以获取上下文向量
        hidden, cell = model.encoder(src_tensor)
        #cell,hidden = model.encoder(src_tensor)
        #cell,hidden = model.encoder(src_tensor)

        context_vector = (hidden[-1] + hidden[-2]).unsqueeze(0)
        # Initialize decoder input with SOS token
        # 使用 SOS 令牌初始化解码器输
        trg_indexes = [EN.vocab.stoi['<sos>']]

        # Loop until max length or EOS token reached
        # 循环直到达到最大长度或 EOS 代币
        for i in range(MAX_LEN):
            trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
            with torch.no_grad():
                #input, last_hidden, encoder_outputs
                output, hidden, cell = model.decoder(
                    trg_tensor,context_vector, hidden
                )
            pred_token = output.argmax(1).item()
            trg_indexes.append(pred_token)

            # Stop looping if EOS token reached
            if pred_token == EN.vocab.stoi['<eos>']:
                break

        # Convert predicted token indexes to text and return
        trg_tokens = [EN.vocab.itos[i] for i in trg_indexes]
        return ' '.join(trg_tokens[1:])


if __name__ == "__main__":
    # Load saved model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    saved_model_path = './.save/myseq2seq_6.pt'  # path to saved model
    model = torch.load(saved_model_path, map_location=device)

    # Load datasets
    batch_size = 32
    train_iter, val_iter, test_iter, DE, EN = load_dataset(batch_size)

    # Translate some example sentences
    examples = ['ich liebe dich', 'ich bin ein berliner', 'guten tag','Ein Mann','Guten Morgen','Wie heißt dieser Ort']
    #examples = ['hello', '你是谁', '你来自哪里', '我是一个中国人']
    for example in examples:
        translation = translate(model, example, DE, EN)
        print(f'{example} -> {translation}')
