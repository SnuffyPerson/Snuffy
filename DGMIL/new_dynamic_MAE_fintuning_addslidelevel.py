import argparse
import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
from torch import nn
import wandb

from model import Linear_projection
from pos_score_calculator import get_score_and_dis_MAE, get_score_and_dis_feats, get_auc_score
from preprocessor import main_MAE_generator, main_generator_dynamic, new_main_MAE_generator
from data_handler import generate_data

parser = argparse.ArgumentParser(description="DG-MIL main")

parser.add_argument("--exp-name", type=str, default="DG-MIL fintuning and test per epoch")
# parser.add_argument("--dis_training_neg", type=str, default='./MAE_dynamic_trainingneg_dis.npy')
# parser.add_argument("--dis_training_pos", type=str, default='./MAE_dynamic_trainingpos_dis.npy')
# parser.add_argument("--feats_training_neg", type=str, default='./MAE_dynamic_trainingneg_feats.npy')
# parser.add_argument("--feats_training_pos", type=str, default='./MAE_dynamic_trainingpos_feats.npy')
# parser.add_argument("--neg_dir_training", type=str, default='./Cam16_training_neg_features.npy')
# parser.add_argument("--pos_dir_training", type=str, default='./Cam16_training_pos_features.npy')
parser.add_argument("--neg_dir_testing", type=str, default='./MAE_testing_neg_feats.npy')
parser.add_argument("--pos_dir_testing", type=str, default='./MAE_testing_pos_feats.npy')

parser.add_argument("--model_save_dir", type=str, default='./MAE_dynamic_fintuning/')
parser.add_argument("--epoch", type=int, default=50)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--num_cluster", type=int, default=10)
parser.add_argument("--seed", type=int, default=2022)
parser.add_argument('--summary_name', type=str, default='DGMIL_MAE_dynamic_cluster10_')
parser.add_argument("--batch_size", type=int, default=5)
parser.add_argument("--train_frac", type=float, default=0.95)

# for slide testing
parser.add_argument("--testing_feats_original", type=str, default='./test_MAE_feats.npy')
parser.add_argument("--num_bag_list_index", type=str, default='./num_bag_list_index.npy')
parser.add_argument("--test_slide_label", type=str, default='./test_slide_label.npy')

args = parser.parse_args()

writer = SummaryWriter(comment=args.summary_name)

if not os.path.isdir(args.model_save_dir):
    os.mkdir(args.model_save_dir)


def print_(loss):
    wandb.log({'train_loss': loss})
    print(f'\tThe loss calculated: {loss}')


def model_train(train_feat, train_label, model, loss_fn, optimizer):
    model.train()
    y_pred = model(train_feat)
    train_label = train_label.to(device)
    loss = loss_fn(y_pred, train_label)
    print_(loss.item())

    # Zero gradients
    optimizer.zero_grad()
    loss.backward()  # Gradients
    optimizer.step()  # Update
    return loss.item()


def model_test(model, val_feat, labels_test, name):
    model.eval()
    pred = model(val_feat)
    pred = pred.detach().cpu().numpy()
    a, b = accuracy_score(labels_test, np.argmax(pred, axis=1)), roc_auc_score(labels_test, np.argmax(pred, axis=1))
    wandb.log({f'accuracy of extreme samples {name}': a, f'auc of extreme samples {name}': b})
    print('\tThe accuracy of extreme samples test set is', a)
    print('\tThe auc of extreme samples test set is', b)
    return accuracy_score(labels_test, np.argmax(pred, axis=1)), roc_auc_score(labels_test, np.argmax(pred, axis=1))


def model_newfeats_extract(model, feats):
    model.eval()
    new_feats = model.projection_head(feats)
    return new_feats


def change_format_for_feats(feats):
    feats = torch.from_numpy(feats.astype(np.float32))
    return feats.to(device)


def change_format_for_labels(labels):
    labels = torch.from_numpy(labels.astype(np.compat.long)).long()
    return labels


if __name__ == "__main__":
    device = "cuda:0"

    torch.set_default_device(device)

    wandb.init(
        # set the wandb project where this run will be logged
        project="DGMIL Batched",

        # track hyperparameters and run metadata
        config={
            "learning_rate": args.lr,
            "architecture": "DGMIL Batched",
            "dataset": "C16",
            "epochs": args.epoch,
            "num_cluster": args.num_cluster,
            "batch_size": args.batch_size,
            "train_frac": args.train_frac
        }
    )

    negative_data_loader, positive_data_loader = generate_data(train_frac=args.train_frac, batch_size=args.batch_size)

    # loading extreme training samples based on distance (original MAE feats space), note that the feats are not dis,
    # but are picked based on dis (extreme samples) i.e. MAE feats space initialization

    # fintuning_feats, label = main_MAE_generator(args.dis_training_neg, args.dis_training_pos, args.feats_training_neg,
    #                                             args.feats_training_pos)
    # fintuning_feats, label = new_main_MAE_generator(args.feats_training_neg, args.feats_training_pos)
    # features_train, features_test, labels_train, labels_test = train_test_split(fintuning_feats, label, test_size=0.1,
    #                                                                             random_state=12345,
    #                                                                             shuffle=True)
    #
    # # change format for training and testing
    #
    # train_feat = change_format_for_feats(features_train)
    # train_label = change_format_for_labels(labels_train)
    # val_feat = change_format_for_feats(features_test)
    # val_label = change_format_for_labels(labels_test)
    #
    # # loading original testing samples directly from orginal patch features
    #
    # neg_feats_training = np.load(args.feats_training_neg)
    # pos_feats_training = np.load(args.feats_training_pos)
    neg_feats_testing = np.load(args.neg_dir_testing)
    pos_feats_testing = np.load(args.pos_dir_testing)
    #
    # # change format for directly testing all original feats
    #
    all_test_original_feats = np.vstack((neg_feats_testing, pos_feats_testing))
    all_test_original_label = np.array(
        [0] * len(neg_feats_testing) + [1] * len(pos_feats_testing))

    all_test_original_feats = change_format_for_feats(all_test_original_feats)
    all_test_original_label = change_format_for_labels(all_test_original_label)

    # all_train_neg_original_feats = change_format_for_feats(neg_feats_training)
    # all_train_pos_original_feats = change_format_for_feats(pos_feats_training)
    all_neg_feats_testing = change_format_for_feats(neg_feats_testing)
    all_pos_feats_testing = change_format_for_feats(pos_feats_testing)

    # for model and loss initilization


    # loss2 = nn.MSELoss()
    model = Linear_projection()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, args.epoch + 1):
        print("Epoch #", epoch)
        # this for loop reads data from both DataLoaders in parallel
        for batch, (train_pos, train_neg) in enumerate(zip(positive_data_loader, negative_data_loader)):
            print(f'batch={batch}')
            wandb.log({'batch': batch})
            
            new_train_neg_original_feats = model.projection_head(change_format_for_feats(train_neg)).detach().cpu().numpy()
            new_train_pos_original_feats = model.projection_head(change_format_for_feats(train_pos)).detach().cpu().numpy()

            dis_neg_train, dis_pos_train = get_score_and_dis_feats(args.num_cluster, args.seed,
                                                                   np.copy(new_train_neg_original_feats),
                                                                   new_train_pos_original_feats)

            del new_train_pos_original_feats

            new_neg_feats_testing = model.projection_head(all_neg_feats_testing).detach().cpu().numpy()
            new_pos_feats_testing = model.projection_head(all_pos_feats_testing).detach().cpu().numpy()

            aucscore = get_score_and_dis_MAE(args.num_cluster, args.seed, new_train_neg_original_feats,
                                             new_neg_feats_testing,
                                             new_pos_feats_testing)

            del new_train_neg_original_feats, new_neg_feats_testing, new_pos_feats_testing

            fintuning_feats, label = main_generator_dynamic(dis_neg_train, train_neg, dis_pos_train, train_pos)
            features_train, features_test, labels_train, labels_test = train_test_split(fintuning_feats, label,
                                                                                        random_state=12345,
                                                                                        test_size=0.1,
                                                                                        shuffle=True)

            train_feat = change_format_for_feats(features_train)
            del features_train
            train_label = change_format_for_labels(labels_train)
            del labels_train
            val_feat = change_format_for_feats(features_test)
            del features_test
            # val_label = change_format_for_labels(labels_test)

            loss_epoch = model_train(train_feat, train_label, model, loss_fn, optimizer)
            val_acc, val_auc = model_test(model, val_feat, labels_test, 'validation')
            orig_acc, orig_auc = model_test(model, all_test_original_feats, all_test_original_label, 'test')

            writer.add_scalar('Loss_training', loss_epoch, epoch)
            writer.add_scalar('Acc_extreme_samples_testset', val_acc, epoch)
            writer.add_scalar('AUC_extreme_samples_testset', val_auc, epoch)
            writer.add_scalar('Acc_orig_all_feats_directly_using_pretrained_classifier', orig_acc, epoch)
            writer.add_scalar('AUC_orig_all_feats_directly_using_pretrained_classifier', orig_auc, epoch)

            writer.add_scalar('AUC_orig_all_feats_using_new_feats_and_ood_based', aucscore, epoch)

            print(aucscore)

        # for slide-level testing AUC

        testing_feats_original = np.load(args.testing_feats_original)
        testing_feats_original = change_format_for_feats(testing_feats_original)
        new_testing_feats_ = model.projection_head(testing_feats_original).detach().cpu().numpy()
        num_bag_list_index = np.load(args.num_bag_list_index)
        test_slide_label = np.load(args.test_slide_label)

        dis_new_testing_feats, dis_new_testing_feats_ = get_score_and_dis_feats(args.num_cluster, args.seed,
                                                                                new_testing_feats_,
                                                                                new_testing_feats_)
        del dis_new_testing_feats_
        slide_score_all = []
        slide_label_all = []
        for i in range(len(test_slide_label)):
            if i < len(test_slide_label) - 1:
                start = num_bag_list_index[i]
                end = num_bag_list_index[i + 1]
            if i == len(test_slide_label) - 1:
                start = num_bag_list_index[i]
                end = len(dis_new_testing_feats)
            slide_score = np.mean(dis_new_testing_feats[start:end])
            if np.isnan(slide_score):
                continue    # !!!!!!!!!!!!!!! Is there a better way?
            slide_score_all.append(slide_score)
            if 'p' in test_slide_label[i]:
                slide_label_all.append(1)
            else:
                slide_label_all.append(0)

        slide_score_all = np.array(slide_score_all)
        slide_label_all = np.array(slide_label_all)
        slide_auc = get_auc_score(slide_label_all, slide_score_all)

        print(slide_auc)
        writer.add_scalar('Slide_AUC_using_new_feats_and_ood_based', slide_auc, epoch)

        wandb.log({'slide_auc': slide_auc, 'epoch': epoch})

        print("")
        del testing_feats_original, new_testing_feats_, num_bag_list_index, test_slide_label, dis_new_testing_feats,\
            slide_score_all, slide_label_all, slide_auc

    print("Finish!")
    wandb.finish()
