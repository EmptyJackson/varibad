import os
import wandb
import datetime

class WandbLogger:
    def __init__(self, args, exp_label):
        # Initialise Weights and Biases
        if args.wandb_id == None:
            self.wandb_id = wandb.util.generate_id()
        else:
            self.wandb_id = args.wandb_id

        run = wandb.init(entity=args.wandb_entity,
                         project=args.wandb_project,
                         group=args.wandb_group,
                         job_type=args.wandb_job_type,
                         save_code=True,
                         settings=wandb.Settings(start_method='fork'),
                         id=self.wandb_id,
                         resume='allow')
        wandb.config.update(args, allow_val_change=True)

        self.resumed = wandb.run.resumed
        if self.resumed:
            print('Resuming wandb run', self.wandb_id)

        # Create output folder for visualisations
        self.output_name = exp_label + '_' + self.wandb_id
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

        self.entry = {}

    def add(self, name, value, x_pos):
        if name in self.entry:
            wandb.log(self.entry)
            self.entry = {}
        self.entry[name] = value
