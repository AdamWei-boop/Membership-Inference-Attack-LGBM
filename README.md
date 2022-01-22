# Membership Inference Attack
Membership inference attack on neural networks with MNIST and CIFAR10 datasets

    parser = argparse.ArgumentParser()
    
    # inference type
    parser.add_argument('--is_target_train', type=bool, default=True, help="Whether training target model")    
    parser.add_argument('--is_DP', type=bool, default=True, help="Whether training target model with DP")    
    
    # path:
    parser.add_argument('--model_target_dir', type=str, default='.')
    parser.add_argument('--data_train_dir', type=str, default='.')
    parser.add_argument('--data_val_dir', type=str, default='.')
    parser.add_argument('--data_train_shadow_dir', type=str, default='.')
    parser.add_argument('--data_val_shadow_dir', type=str, default='.')            
    
    # general:
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train_target_size', type=int, default=2500, help='2 500, 5 000, 10 000, 15 000 // 4 600, 10 520, 19 920, 29 540')
    parser.add_argument('--test_target_size', type=int, default=1000)
    parser.add_argument('--number_shadow_model', type=int, default=25, help='25 50 MNIST and 100 cifar')
    
    # learning:
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--decrease_lr_factor', type=float, default=1e-7)
    parser.add_argument('--decrease_lr_every', type=int, default=1)
    
    parser.add_argument('--reg_lambd', type=int, default=10)
    parser.add_argument('--n_estimators', type=int, default=10000)
        
    # statistics:
    parser.add_argument('--dataset', type=str, default='MNIST', help="MNIST, CIFAR10, CIFAR100")
    parser.add_argument('--training_size_value', type=list, default=[4600, 10520, 19920, 29540], help="[2500, 5000, 10000, 15000]")
    parser.add_argument('--number_shadow_value', type=list, default=[100], help='[2, 10 ,20, 50, 100]')
    parser.add_argument('--epoch_value', type=list, default=[100], help="[2, 10 ,20, 50, 100]")
    parser.add_argument('--experments', type=int, default=3)
        
    # differential privacy
    parser.add_argument('--eps', type=int, default=2)    
    parser.add_argument('--set_eps', type=list, default=[1,20,100])    
    
    args = parser.parse_args()
    
    
    # Example 1: training target model with various DP levels
    if args.is_target_train:
        
        acc_eps = []
        
        for args.eps in args.set_eps:
            acc_list = []
            for i in range(args.experments):
            
                precision_general, recall_general, accuracy_general, precision_per_class,\
                    recall_per_class, accuracy_per_class = mi(args)
                acc_list.append(accuracy_general)
                print('\nEps:', args.eps, 'Attack acc:', accuracy_general)    
            acc_eps.append(sum(acc_list)/len(acc_list))
        print('\nAttack acc list:', acc_eps)
        
    # Example 2: attack the trained model with allocated datasets   
    else:
    
        set_idx = [0,0,0,0]
        set_eps = [2,5,10,20]
        
        acc_eps = []
    
        for i in range(len(set_idx)):
            args.model_target_dir = './mi_attack/model-client{}_eps{}_delta0.001.ckpt'.format(set_idx[i], set_eps[i])
            args.data_train_dir = './mi_attack/dict_train-mnist.pt'
            args.data_val_dir = './mi_attack/dict_test-mnist.pt'
            args.data_train_shadow_dir = './mi_attack/dict_train_shadow-mnist.pt'
            args.data_val_shadow_dir = './mi_attack/dict_test_shadow-mnist.pt'
        
            acc_list = []
            for k in range(args.experments):
                precision_general, recall_general, accuracy_general, precision_per_class,\
                    recall_per_class, accuracy_per_class = mi(args)
                acc_list.append(accuracy_general)
                print('\nEps:', set_eps[i],'Accuracy:', accuracy_general)
            acc_eps.append(sum(acc_list)/len(acc_list))
        print('\nAttack acc list:', acc_eps)
