def train_validate_model(model, train_loader, val_loader,
                criterion, optimizer, EPOCHS):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    best_val_loss = 999.0

    # Keep track of accuracies and losses
    train_losses = []
    val_losses = []
    val_accuracies = []

    # Keep track of the scores
    f1_normal_scores = []
    f1_pneumonia_scores = []
    f1_weighted_scores = []
    f1_macro_scores = []


    print(f"TRAINING BEGINS NOW\n\n")
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.RandomCrop(256, padding=10)
    ])

    best_model = None

    for epoch in range(EPOCHS):

        train_loss = 0.0
        val_loss = 0.0

        val_accuracy = 0.0

        model.train()
        for i, (data, labels) in enumerate(train_loader):
            data = train_transforms(data)
            data = data.to(device)
            labels = labels.to(device).type(torch.long).squeeze(1)

            optimizer.zero_grad()

            outputs = model(data)

            loss = criterion(outputs, labels)

            loss.cpu().backward()

            optimizer.step()

            train_loss += loss.to('cpu').item()

        model.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        total = 0
        correct = 0

        f1_normal = 0.0
        f1_pneumonia = 0.0
        f1_score_weighted_score = 0.0
        f1_score_macro_score = 0.0

        with torch.no_grad():

            for i, (data, labels) in enumerate(val_loader):

                labels = labels.squeeze(1)

                data = data.to(device)
                labels = labels.to(device).type(torch.long)

                outputs = model(data)

                loss = criterion(outputs, labels)

                val_loss += loss.cpu().item()

                # Calculate the correctly classified images and the total
                total += labels.size(0)
                predicted = torch.max(outputs.data, 1)[1]

                correct += (predicted == labels).sum().item()

                val_accuracy += correct/total

                # Calculate f1-score
                f1_score_class_wise = multiclass_f1_score(predicted, labels, num_classes=2, average=None)
                f1_score_weighted = multiclass_f1_score(predicted, labels, num_classes=2, average='weighted')
                f1_score_macro = multiclass_f1_score(predicted, labels, num_classes=2, average='macro')

                # Accumulate the weighted f1-score
                f1_score_weighted_score += f1_score_weighted.item()

                # Accumulate the macro f1-score
                f1_score_macro_score += f1_score_macro.item()

                # Add the f1-scores ASSUME: 'tensor([0., 1.], device='cuda:0')'
                f1_normal += f1_score_class_wise[0].item()
                f1_pneumonia += f1_score_class_wise[1].item()


        avg_f1_score_weighted = f1_score_weighted_score/len(val_loader)
        avg_f1_score_macro = f1_score_macro_score/len(val_loader)

        avg_f1_normal = f1_normal/len(val_loader)
        avg_f1_pneumonia = f1_pneumonia/len(val_loader)

        avg_val_accuracy = val_accuracy/len(val_loader)
        avg_train_loss = train_loss/len(train_loader)
        avg_val_loss = val_loss/len(val_loader)

        # Append the losses and accuracies, and scores
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(avg_val_accuracy)
        f1_normal_scores.append(avg_f1_normal)
        f1_pneumonia_scores.append(avg_f1_pneumonia)
        f1_weighted_scores.append(avg_f1_score_weighted)
        f1_macro_scores.append(avg_f1_score_macro)



        print(f"Epoch: {epoch+1}/{EPOCHS}",
              f"Train Loss: {avg_train_loss}",
              f"Val Loss: {avg_val_loss}",
              f"Val Accuracy: {avg_val_accuracy}",
              f"f1-score_normal: {avg_f1_normal}",
              f"f1-score_pneumonia: {avg_f1_pneumonia}",
              f"f1-score_weighted: {avg_f1_score_weighted}",
              f"f1-score_macro: {avg_f1_score_macro}",
              sep="\n")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

            # copy.Deepcopy the model
            best_model = copy.deepcopy(model)


    return (best_model, train_losses, val_losses, val_accuracies,
            f1_normal_scores, f1_pneumonia_scores, f1_weighted_scores, f1_macro_scores)