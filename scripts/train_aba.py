import os
import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from spiraton.data.aba_dataset import ABADataset
from spiraton.training.aba_model import ABAModel
from spiraton.training.loss import ABALoopLoss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, default="../corpus_eve_clean.txt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--use_chrono", action="store_true", help="Utiliser la cellule ChronoSpiraton (second ordre)")
    parser.add_argument("--pure_phonetic", action="store_true", help="Alimenter directement avec les dimensions 8-22")
    args = parser.parse_args()
    
    corpus_path = os.path.abspath(args.corpus)
    if not os.path.exists(corpus_path):
        print(f"Corpus non trouvé : {corpus_path}")
        return
        
    dataset = ABADataset(corpus_path, max_tokens_per_segment=16)
    if len(dataset) == 0:
        print("Aucun cycle valide trouvé dans le corpus.")
        return
        
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    model = ABAModel(dim=64, state_size=64, use_chrono=args.use_chrono, pure_phonetic=args.pure_phonetic)
    loss_fn = ABALoopLoss(copy_penalty_weight=0.1, supervised_weight=1.0)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    print(f"Début de l'entraînement sur {len(dataset)} cycles ABA...")
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        
        for batch in dataloader:
            optimizer.zero_grad()
            
            a_vecs = batch["A"]
            b_vecs = batch["B"]
            a_prime_vecs = batch["A_PRIME"]
            
            a_prime_pred, a_agg = model(a_vecs, b_vecs)
            
            if args.pure_phonetic:
                a_prime_emb = a_prime_vecs[..., 8:23]
            else:
                a_prime_emb = model.embeddings(a_prime_vecs)
                
            a_prime_true = a_prime_emb.mean(dim=1).detach()
            
            loss = loss_fn(a_prime_pred, a_agg.detach(), a_prime_true)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {total_loss/len(dataloader):.4f}")
        
    print("Entraînement terminé avec succès.")

if __name__ == "__main__":
    main()
