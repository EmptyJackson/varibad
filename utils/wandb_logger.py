import os
import wandb
import datetime

class WandbLogger:
    def __init__(self, args, exp_label):
        # Create output folder for visualisations
        self.output_name = exp_label + '_' + str(args.seed) + '_' + datetime.datetime.now().strftime('_%d:%m_%H:%M:%S')
        try:
            log_dir = args.results_log_dir
        except AttributeError:
            log_dir = args['results_log_dir']

        if log_dir is None:
            dir_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
            dir_path = os.path.join(dir_path, 'logs')
        else:
            dir_path = log_dir

        if not os.path.exists(dir_path):
            try:
                os.mkdir(dir_path)
            except:
                dir_path_head, dir_path_tail = os.path.split(dir_path)
                if len(dir_path_tail) == 0:
                    dir_path_head, dir_path_tail = os.path.split(dir_path_head)
                os.mkdir(dir_path_head)
                os.mkdir(dir_path)

        try:
            self.full_output_folder = os.path.join(os.path.join(dir_path, 'logs_{}'.format(args.env_name)),
                                                   self.output_name)
        except:
            self.full_output_folder = os.path.join(os.path.join(dir_path, 'logs_{}'.format(args["env_name"])),
                                                   self.output_name)

        if not os.path.exists(self.full_output_folder):
            os.makedirs(self.full_output_folder)
        
        # Initialise Weights and Biases
        run = wandb.init(entity=args.wandb_entity,
                         project=args.wandb_project,
                         group=args.wandb_group,
                         job_type=args.wandb_job_type,
                         save_code=True,
                         settings=wandb.Settings(start_method='fork'))
        wandb.config.update(args)

    def add(self, name, value, x_pos):
        wandb.log({name: value})
