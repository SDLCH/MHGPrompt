from torchmetrics.classification import MulticlassF1Score
import torch


def evaluate(feats, mps, idx_test, label, model, prompt, answering, num_class, device):
    prompt.eval()
    answering.eval()

    macro_f1 = MulticlassF1Score(num_classes=num_class, average='macro').to(device)
    micro_f1 = MulticlassF1Score(num_classes=num_class, average='micro').to(device)

    with torch.no_grad():
        feat = prompt.add(feats)
        embeds = model.get_embeds(feat, mps)
        test_embs = embeds[idx_test]
        test_lbls = torch.argmax(label[idx_test], dim=-1)
        out = answering(test_embs)
        pred = out.argmax(dim=1)

        ma_f1 = macro_f1(pred, test_lbls)
        mi_f1 = micro_f1(pred, test_lbls)

        print(f"Macro-F1: {ma_f1:.4f}| Micro-F1: {mi_f1:.4f}")

    return ma_f1.item(), mi_f1.item()
