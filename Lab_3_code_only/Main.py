import gc
import os
import json
import io
import queue
import threading
import time

import PIL
import dill


import numpy as np
import pathlib

import torch
import torchvision.transforms.transforms
from torch.utils.data import DataLoader, TensorDataset
import torch.nn
from torch.optim import Adam
from torchvision.transforms import transforms

from Data_preprocess import caption_preprocess, word_embedding
from Model import multimodal_model
import gc as garbage_man

# Main task: Preprocess data, define model, and train
def Task1():
    # Run data preprocessing
    # caption_preprocess()  # Extract and preprocess text and images
    
    max_length = 35
    truncation_length = 50
    end_token_idx = 15

    data_tuple_path = pathlib.Path('./data_tuple.dill')
    if data_tuple_path.exists():
        with open(data_tuple_path, 'rb') as file:
            image_train, image_test, target_caption_train, input_caption_train, target_caption_test, input_caption_test, input_caption_pred, emb_mat = dill.load(file)
    else:
        data_tuple = word_embedding(max_length)  # Load preprocessed data
        garbage_man.collect()
        with open(data_tuple_path, 'wb') as file:
            dill.dump(data_tuple, file)

        image_train, image_test, target_caption_train, input_caption_train, target_caption_test, input_caption_test, input_caption_pred, emb_mat = data_tuple
        del data_tuple
        garbage_man.collect()

    # Model parameters
    # Original 224
    image_shape = (3, 224, 224)
    vocab_size, emb_dim = np.shape(emb_mat)
    
    # Define multimodal model
    model = multimodal_model(max_length, vocab_size, image_shape, emb_dim, emb_mat, end_token_idx, truncation_length)
    model.remove_softmax()

    training_loader = DataLoader(
        TensorDataset(torch.tensor(image_train), torch.tensor(input_caption_train), torch.tensor(target_caption_train)),
        batch_size=64,
        shuffle=True
    )

    test_loader = DataLoader(
        TensorDataset(torch.tensor(image_test), torch.tensor(input_caption_test), torch.tensor(target_caption_test)),
        batch_size=64,
        shuffle=False
    )

    # criterion = torch.nn.NLLLoss(ignore_index=0)
    transform = transforms.Compose([
        # 128 x 128
        transforms.Resize((224, 224), antialias=False),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Example normalization
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224), antialias=False),
    ])
    resize = transforms.Resize((224, 224), antialias=False)

    from torchinfo import summary
    summary(model, input_size=(32, 3, 224, 224), device='cuda' if torch.cuda.is_available() else 'cpu')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    start_time = time.time()
    seq_start = time.time()

    print("Model is set up")

    # Train model.CNN as an autoencoder first to get some initial image features
    # Then train the multimodal model
    image_training_loader = DataLoader(
        TensorDataset(torch.tensor(image_train)),
        batch_size=64,
        shuffle=True
    )

    image_test_loader = DataLoader(
        TensorDataset(torch.tensor(image_test)),
        batch_size=64,
        shuffle=False
    )

    class tensor_3d_view(torch.nn.Module):
        def __init__(self, C, H, W, *args, **kwargs):
            super(tensor_3d_view, self).__init__(*args, **kwargs)
            self.C = C
            self.H = H
            self.W = W

        def forward(self, x):
            # Take flattened features and reshape to (N, C, W, H)
            return x.view(-1, self.C, self.W, self.H)

    """# Move model to CPU, then move CNN back to GPU
    model = model.to('cpu')
    torch.save(model.CNN, 'cnn.pth')
    CNN = torch.load('cnn.pth')

    CNN = CNN.to('cuda' if torch.cuda.is_available() else 'cpu')

    auto_enc = torch.nn.Sequential(
        CNN,
        # Output shape is (N, 300)
        # Use CNNs to decode the image and expand it to the original shape (3, 224, 224)
        torch.nn.Linear(300, 112 * 112 * 3),
        torch.nn.LeakyReLU(0.2, True),
        tensor_3d_view(3, 112, 112),
        torch.nn.ConvTranspose2d(3, 16, 3, padding=1),
        torch.nn.BatchNorm2d(16),
        torch.nn.LeakyReLU(0.2, True),
        torch.nn.ConvTranspose2d(16, 3, 3, padding=1),
        torch.nn.BatchNorm2d(3),
        torch.nn.LeakyReLU(0.2, True),
    )

    summary(auto_enc, input_size=(32, 3, 224, 224), device='cuda' if torch.cuda.is_available() else 'cpu')
    cnn_optimizer = Adam(auto_enc.parameters(), lr=1e-3, weight_decay=1e-5)
    cnn_criterion = torch.nn.MSELoss()
    import datetime
    for epoch in range(5):
        total = 0
        seq_total = 0
        splice_total = 0

        e_loss = 0
        correct = 0
        # Timestamp each epoch (DD/MM/YYYY HH:MM:(SS.1f)
        timestamp = datetime.datetime.now()

        print(f"\n__ Epoch {epoch + 1} __ | {timestamp.year:4d}/{timestamp.month:02d}/{timestamp.day:02d} {timestamp.hour:02d}:{timestamp.minute:02d}:{timestamp.second:02d}.{timestamp.microsecond // 100000} __")

        auto_enc.train()
        for i, image in enumerate(image_training_loader):
            # Image is N, W, H, C
            image = image[0].to(device)
            # Transform expects N, C, W, H
            image = image.permute(0, 3, 1, 2)
            image = transform(image.float() / 255)
            cnn_optimizer.zero_grad()
            output = auto_enc(image)
            output = resize(output)
            loss = cnn_criterion(output, image)
            loss: torch.Tensor
            e_loss += loss.item()
            loss.backward()
            cnn_optimizer.step()

            if time.time() - start_time > 1 / 4:
                progress_bar = "=" * int(59 * (i + 1) / len(image_training_loader)) + ">"
                progress_bar = progress_bar.ljust(60, ' ')
                print(f"Train| Epoch: {epoch}, Batch: {i + 1}/{len(image_training_loader)}, |{progress_bar}|, Loss: {e_loss / len(image_training_loader):.4g}\033[0K", end='\r', flush=True)
                start_time = time.time()

        progress_bar = "=" * 60
        print(f"Train| Epoch: {epoch}, Batch: {len(image_training_loader)}/{len(image_training_loader)}, |{progress_bar}|, Loss: {e_loss / len(image_training_loader):.4g}\033[0K")

        print("\n", end="")
        total = 0
        seq_total = 0
        splice_total = 0
        e_loss = 0
        correct = 0
        auto_enc.eval()
        with torch.no_grad():
            for i, image in enumerate(image_test_loader):
                # Image is W, H, C
                image = image[0].to(device)
                image = image.permute(0, 3, 1, 2)
                image = test_transform(image.float() / 255)
                output = auto_enc(image)
                output = resize(output)
                loss = cnn_criterion(output, image)
                e_loss += loss.item()

                if time.time() - start_time > 1 / 4:
                    progress_bar = "=" * int(59 * (i + 1) / len(image_test_loader)) + ">"
                    progress_bar = progress_bar.ljust(60, ' ')
                    print(f"Eval| Epoch: {epoch}, Batch: {i + 1}/{len(image_test_loader)}, |{progress_bar}|, Loss: {e_loss / len(image_test_loader):.4g}\033[0K", end='\r', flush=True)
                    start_time = time.time()
        progress_bar = "=" * 60
        print(f"Eval| Epoch: {epoch}, Batch: {len(image_test_loader)}/{len(image_test_loader)}, |{progress_bar}|, Loss: {e_loss / len(image_test_loader):.4g}\033[0K")
        print("\n", end="")

    print("CNN training complete")

    torch.save(auto_enc[0], 'cnn.pth')
    import gc
    del auto_enc
    gc.collect()
    torch.cuda.empty_cache()"""

    cnn = torch.load('cnn.pth')
    model.CNN = cnn
    model.to(device)

    model = torch.load('model.pth')
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='sum')

    """ascii_move_up = "\033[1A"
    import datetime
    for epoch in range(50):
        total = 0
        seq_total = 0
        splice_total = 0

        e_loss = 0
        correct = 0
        # Timestamp each epoch (DD/MM/YYYY HH:MM:(SS.1f)
        timestamp = datetime.datetime.now()

        print(f"\n__ Epoch {epoch + 1} __ | {timestamp.year:4d}/{timestamp.month:02d}/{timestamp.day:02d} {timestamp.hour:02d}:{timestamp.minute:02d}:{timestamp.second:02d}.{timestamp.microsecond // 100000} __")

        model.train()
        for i, (image, input_caption, target_caption) in enumerate(training_loader):
            # Image is W, H, C
            image = image.to(device)
            input_caption = input_caption.to(device)
            target_caption = target_caption.to(device)

            image = image.permute(0, 3, 1, 2)
            image = transform(image.float() / 255)
            total += target_caption.size(0)
            seq_total += target_caption.size(1) * target_caption.size(0)

            print("\n", end='', flush=True)
            shuffled_range = range(1, target_caption.size(1))
            shuffled_range = np.random.permutation(shuffled_range)
            # Shuffle to keep stochastic sampling
            for j, idx in enumerate(shuffled_range):
                optimizer.zero_grad()
                if target_caption[:, idx].eq(0).all():
                    # Avoid nan loss
                    continue

                if time.time() - seq_start > 1 / 4:
                    seq_progress_bar = "-" * int(29 * (j + 1) / target_caption.size(1)) + ">"
                    seq_progress_bar = seq_progress_bar.ljust(30, ' ')
                    print(f"Sequence Progress: |{seq_progress_bar}|\033[0K", end='\r', flush=True)

                optimizer.zero_grad()
                input_caption_splice = input_caption[:, :idx + 1]
                output = model(image, input_caption_splice, 'train')
                loss = criterion(output[:, idx, :], target_caption[:, idx].type(torch.long))
                loss: torch.Tensor

                prediction = torch.argmax(output, dim=2)

                mask = prediction.type(torch.long) == target_caption[:, :idx + 1].type(torch.long)
                mask: torch.Tensor
                correct += torch.sum(mask).item()
                splice_total += mask.numel()
                e_loss += loss.item()
                loss.backward()
                optimizer.step()

            seq_progress_bar = "-" * 30
            print(f"Sequence Progress: |{seq_progress_bar}|\033[0K", end='\r', flush=True)
            print(ascii_move_up, end='\r', flush=True)

            if time.time() - start_time > 1 / 4:
                splice_total = splice_total if splice_total else float("inf")
                progress_bar = "=" * int(59 * (i + 1) / len(training_loader)) + ">"
                progress_bar = progress_bar.ljust(60, ' ')
                print(f"Train| Epoch: {epoch}, Batch: {i + 1}/{len(training_loader)}, |{progress_bar}|, Loss: {e_loss / seq_total:.4g} ({np.exp(e_loss / seq_total):.4g}), Accuracy: {100 * correct / splice_total:.4f}%\033[0K", end='\r', flush=True)
                start_time = time.time()

        progress_bar = "=" * 60
        print(f"Train| Epoch: {epoch}, Batch: {len(training_loader)}/{len(training_loader)}, |{progress_bar}|, Loss: {e_loss / seq_total:.4g} ({np.exp(e_loss/seq_total):.4g}), Accuracy: {100 * correct / splice_total:.2f}%\033[0K")

        print("\n", end="")
        total = 0
        seq_total = 0
        splice_total = 0
        e_loss = 0
        correct = 0
        model.eval()
        with torch.no_grad():
            for i, (image, input_caption, target_caption) in enumerate(test_loader):
                # Image is W, H, C
                image = image.to(device)
                input_caption = input_caption.to(device)
                target_caption = target_caption.to(device)

                image = image.permute(0, 3, 1, 2)
                image = test_transform(image.float() / 255)
                total += target_caption.size(0)
                seq_total += target_caption.size(1) * target_caption.size(0)

                print("\n", end='', flush=True)
                for idx in range(1, target_caption.size(1)):
                    if time.time() - seq_start > 1 / 4:
                        seq_progress_bar = "-" * int(29 * (idx + 1) / target_caption.size(1)) + ">"
                        seq_progress_bar = seq_progress_bar.ljust(30, ' ')
                        print(f"Sequence Progress: |{seq_progress_bar}|\033[0K", end='\r', flush=True)

                    if target_caption[:, idx].eq(0).all():
                        # Avoid nan loss
                        break
                    optimizer.zero_grad()
                    input_caption_splice = input_caption[:, :idx + 1]
                    output = model(image, input_caption_splice, 'train')
                    loss = criterion(output[:, idx, :], target_caption[:, idx].type(torch.long))
                    prediction = torch.argmax(output, dim=2)
                    mask = prediction.type(torch.long) == target_caption[:, :idx + 1].type(torch.long)
                    splice_total += mask.numel()
                    correct += torch.sum(mask).item()
                    e_loss += loss.item()

                seq_progress_bar = "-" * 30
                print(f"Sequence Progress: |{seq_progress_bar}|\033[0K", end='\r', flush=True)
                print(ascii_move_up, end='\r', flush=True)


                if time.time() - start_time > 1 / 4:
                    progress_bar = "=" * int(59 * (i + 1) / len(test_loader)) + ">"
                    progress_bar = progress_bar.ljust(60, ' ')
                    print(f"Eval| Epoch: {epoch}, Batch: {i + 1}/{len(test_loader)}, |{progress_bar}|, Loss: {e_loss / seq_total:.4g} ({np.exp(e_loss / seq_total):.4g}), Accuracy: {100 * correct / splice_total:.4f}%\033[0K", end='\r', flush=True)
                    start_time = time.time()
        print(f"Eval| Epoch: {epoch}, Batch: {len(test_loader)}/{len(test_loader)}, |{progress_bar}|, Loss: {e_loss / seq_total:.4g} ({np.exp(e_loss / seq_total):.4g}), Accuracy: {100 * correct / splice_total:.4f}%\033[0K")
        print("\n", end="")"""
    # torch.save(model, 'model.pth')
    print("Training complete")
    test_tensor = torch.tensor(image_test)
    pred_data = DataLoader(
        TensorDataset(test_tensor, torch.tensor([input_caption_pred]*test_tensor.size(0))),
        batch_size=64,
        shuffle=False
    )
    # Put in training data instead of test data

    # image_train = torch.tensor(image_train)
    # pred_data = DataLoader(
    #     TensorDataset(torch.tensor(image_train), torch.tensor([input_caption_pred] * image_train.size(0))),
    #     batch_size=128,
    #     shuffle=True
    # )

    model.eval()

    predictions = []

    with open("word_map.json", "rb") as f:
        word_map = json.loads(f.read())

    reverse_word_map = {v["Rep"]: k for k, v in word_map.items()}
    reverse_word_map[0] = "<pad>"

    with torch.no_grad():
        for i, (image, input_caption) in enumerate(pred_data):
            print(f"Batch {i + 1}/{len(pred_data)}")
            image = image.permute(0, 3, 1, 2)
            image = transform(image.float() / 255)
            image = image.to(device)
            input_caption = input_caption.to(device)
            output = model(image, input_caption, 'infer')
            predictions.append((image.cpu(), output.cpu()))
            gc.collect()
            torch.cuda.empty_cache()


    # List[tensor]
    #   - shape (batch_size, max_length)

    pil_transorm = transforms.ToPILImage()

    prediction_sentences = dict()
    for i, (image_batch, prediction) in enumerate(predictions):
        print(f"Batch {i + 1}/{len(predictions)}")
        prediction_sentences[i] = []
        for j, pred in enumerate(prediction):
            pred = pred.tolist()

            image = image_batch[j].permute(1, 2, 0).detach()
            # Bring image back to 0-255 uint8 range
            image = (image - image.min()) / (image.max() - image.min()) * 255
            image = image.type(torch.uint8).cpu().numpy()
            PIL_image = pil_transorm(image)

            prediction_sentences[i].append({"image": PIL_image, "sentences": []})

            pred = [reverse_word_map[word] for word in pred]
            pred = " ".join(pred)
            prediction_sentences[i][-1]["sentence"] = pred

    preds_path = pathlib.Path("topk3/predictions")
    (preds_path / "batches").mkdir(exist_ok=True, parents=True)
    all_captions = pathlib.Path("topk3/all_captions")
    all_captions.mkdir(exist_ok=True, parents=True)

    image_queue = queue.Queue()
    def handle_tasks(image_queue: queue.Queue, pred_batch_path: pathlib.Path, all_captions: pathlib.Path, print_lock: threading.Lock):
        start_time = time.time()
        while True:
            try:
                item = image_queue.get(block=True, timeout=1)
                if item is None:
                    print(f"T{threading.get_ident()} Sentinel Received, Exiting...")
                    break
                pred, caption_images_saved, i, j = item

                # Save image
                pred["image"].save(pred_batch_path / f"{j}.png")

                while True:
                    try:
                        fig, ax = plt.subplots()
                        ax.imshow(mpimg.imread(pred_batch_path / f"{j}.png"))
                        # Center the text on the image, and place it at the bottom
                        # Wrap text that is over 50 characters
                        ax.text(0.5, 0.05, "-\n".join(pred["sentence"][k:k + 50] for k in range(0, len(pred["sentence"]), 50)),
                                ha='center', va='center', transform=ax.transAxes, fontsize=12, color='white', wrap=True, backgroundcolor='black')

                        ax.axis('off')
                        # Previously, we saved the image using plt.savefig, but we can't do that here
                        # plt.tight_layout()
                        # plt.savefig(pred_batch_path / f"{j}_caption.png")
                        # plt.savefig(all_captions / f"{caption_images_saved}.png")

                        # Use fig and ax to save the image, we don't want to hog plt
                        fig.tight_layout()
                        fig.savefig(pred_batch_path / f"{j}_caption.png")
                        fig.savefig(all_captions / f"{caption_images_saved}.png")
                    except Exception as e:
                        print(f"T{threading.get_ident()} Error: {e.__class__.__name__}: {e}")
                        continue
                    break

                if time.time() - start_time > 1 / 4:
                    with print_lock:
                        print(f"T{threading.get_ident()} Saving Image {caption_images_saved}/{len(predictions) * predictions[0][1].size(0)}")
                    start_time = time.time()
                plt.close(fig)
            except queue.Empty:
                time.sleep(1)
                continue

    print_lock = threading.Lock()
    t = [threading.Thread(target=handle_tasks, args=(image_queue, preds_path / "batches", all_captions, print_lock)) for _ in range(5)]
    for t_start in t:
        t_start.start()
    caption_images_saved = 0
    for i, batch in prediction_sentences.items():
        # Create batch folder
        pred_batch_path = preds_path / "batches" / f"{i}"
        pred_batch_path.mkdir(exist_ok=True, parents=True)

        for j, pred in enumerate(batch):
            # Save text
            with open(pred_batch_path / f"{j}.txt", "w", encoding="utf-8") as f:
                f.write(pred["sentence"])

            # Save white padded image with text caption (bottom)
            # Open the image with matplotlib, add the text, and save it
            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg

            pred["sentence"] = pred["sentence"].replace("endseq", "").replace("<pad>", "").strip()

            image_queue.put((pred, caption_images_saved, i, j))
            caption_images_saved += 1

        print(f"Saved Batch Info: {i + 1}/{len(prediction_sentences)}")
    for _ in range(len(t) * 2):
        image_queue.put(None)
    print(f"Sentinel Sent")
    for t_end in t:
        t_end.join()
    print(f"Image Queue Joined")

    print(*predictions, sep='\n' * 2)
# Entry point for the script
def main():
    Task1()

if __name__ == "__main__":
    main()