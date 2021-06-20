import sys
import os
import hashlib
import json 
from loguru import logger 
from datetime import datetime 
def cstr(arg, arg_name, default, custom_str=False):
    """ Get config str for arg, ignoring if set to default. """
    not_default = arg != default
    if not custom_str: # no custom_str, use name+value 
        custom_str = f'_{arg_name}{arg}'
    return custom_str if not_default else '' # return custom_str if not default value 
#def create_exp_name_prior(cfg):
#    """ create exp name for prior model 
#    """
#    # ---- create folder name ----  
#    output_folder = format_e(cfg.lr)
#    output_folder += 'b%dh%d'%(cfg.batch_size, cfg.hidden_size_prior) 
#    if cfg.loc_loss_weight != 1.0 or cfg.cls_loss_weight != 1.0:
#        output_folder += 'lw%d%d'%(cfg.loc_loss_weight, cfg.cls_loss_weight)
#    output_folder += print_ifnot(cfg.nloc, -1, 'n%d'%cfg.nloc)
#    output_folder += 'p%d'%cfg.pw 
#    ##print_ifnot(cfg.nloc, -1, 'n%d'%cfg.nloc)
#    if cfg.model_name == 'cnn_prior': 
#        output_folder += print_ifnot(cfg.loc_map, 1, '_reg') 
#        output_folder += print_ifnot(cfg.inputd, 2, '_ind1') 
#        output_folder += print_ifnot(cfg.single_sample, 0, '_ss') # single_sample 
#    output_folder += print_ifnot(cfg.add_stop, 0, '_asp') 
#    if cfg.add_stop:
#        assert(cfg.loc_map == 0), 'not support both add_stop and loc_map' 
#        assert(cfg.add_empty == 0), 'not support both add_stop and add_empty -> will be all as non-stop'
#    output_folder += print_ifnot(cfg.use_scheduler, 0, 'sc') # add stop
#    output_folder += print_ifnot(cfg.early_stopping, 0, 'Estp') # add stop
#    output_folder += print_ifnot(cfg.use_emb_enc, 0, 'Enemb') # use nn embedding for encoder head
#
#    prefix = cfg.output_folder # user input folder name's prefix 
#    output_folder = prefix + output_folder 
#
#    # ---- create the root of the exp output_folder ---- 
#
#    datestr = datetime.now().strftime("%m%d")
#    OUT_ROOT = os.path.join('exp', cfg.model_name, cfg.dataset, datestr)
#    return OUT_ROOT, output_folder
        

def create_exp_name(cfg):
    datestr = datetime.now().strftime("%m%d/")
    logger.debug('date: {}', datestr)
    exp_name = datestr + cfg.tag + format_e(cfg.lr)  
    if cfg.use_patch_bank: 
        exp_name += '_K%dw%d'%(cfg.K, cfg.ww)
        exp_name += f'{cstr(cfg.use_prior_model, "usepm", 1)}'

    if cfg.model_name == 'vae': 
        exp_name += '_%s'%(cfg.vae.layers) 
    if cfg.model_name in ['cvae']:
        exp_name += 'H%dZ%d'%(cfg.enc_hid, cfg.latent_d)
    elif cfg.model_name in ['cvaer', 'cvae2']:
        exp_name += 'Z%d'%(cfg.latent_d)

    exp_name += f'{cstr(cfg.optim.diff_lr, None, 0, "_difflr")}'
    exp_name += '_b%d'%(cfg.batch_size) 
    exp_name += f'{cstr(cfg.seed, "s", 0)}'
    return exp_name 
#def quickstop():
#    flag = os.environ.get('FASTRUN')
#    if flag is None:
#        return 0 
#    else:
#        return int(flag)
#def create_dir(p):
#    if not os.path.isdir(p):
#        os.makedirs(p)
#
def get_time():
    datestr = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    return datestr
#
#def check_out_EVAL(entry, **kwargs):
#    # entry = os.path.basename(__file__)
#    EVAL = os.environ.get('EVAL')
#    jid = os.getenv('SLURM_JOB_ID', None) 
#    tid = os.environ.get('SLURM_ARRAY_TASK_ID', None)
#    if not EVAL: 
#        return 1 # run train 
#    main_json = './track/sub/%s/main.json'%(EVAL)
#    de =json.load(open(main_json, 'r')) if os.path.exists(main_json) else {}
#    de_dict = de[entry] if entry in de else {} 
#    de_dict['check_out'] = 1
#    if 'check_out_info' not in de_dict: de_dict['check_out_info'] = {}
#    for k, v in kwargs.items():
#        de_dict['check_out_info'][k] = '{}'.format(v)
#    if jid:
#        de_dict['check_out_info'].update({'jid': jid})
#    if tid:
#        de_dict['check_out_info'].update({'tid': tid, 'jtid': jid+'_'+tid})
#    de[entry] = de_dict 
#    json.dump(de, open(main_json, 'w'), indent=2)
#    logger.info('%s check out %s'%(entry, main_json))
#    return 1
#
#def check_in_EVAL(entry, **kwargs):
#    # entry = os.path.basename(__file__)
#    EVAL = os.environ.get('EVAL')
#    jid = os.getenv('SLURM_JOB_ID', None) 
#    tid = os.environ.get('SLURM_ARRAY_TASK_ID', None)
#    if not EVAL: 
#        return 0 # run train 
#    main_json = './track/sub/%s/main.json'%(EVAL)
#    if not os.path.exists(os.path.dirname(main_json)):
#        os.makedirs(os.path.dirname(main_json))
#        ## json.dump({'EVAL': EVAL}, open(main_json, 'w'), indent=2)
#    de =json.load(open(main_json, 'r')) if os.path.exists(main_json) else {}
#    de_dict = de[entry] if entry in de else {} 
#    exp_done = de_dict.get('check_out') # if 1 
#    if exp_done: # exp done 
#        logger.info('[exp_done] {} | <{}> done for [ {} ] | json: {} ', 
#            de_dict.get('check_out_info'), entry, EVAL, main_json)
#        return 1
#    de_dict['check_in'] = 1
#    if 'check_in_info' not in de_dict: 
#        de_dict['check_in_info'] = {}
#
#    for k, v in kwargs.items():
#        de_dict['check_in_info'][k] = '{}'.format(v)
#
#    if jid:
#        de_dict['check_in_info'].update({'jid': jid})
#    if tid:
#        de_dict['check_in_info'].update({'tid': tid, 'jtid': jid+'_'+tid})
#    de[entry] = de_dict 
#    json.dump(de, open(main_json, 'w'), indent=2)
#    logger.info('%s check in to %s'%(entry, main_json))
#    return 0
#
#def try_write_EVAL(d, writer='main'):
#    EVAL = os.environ.get('EVAL')
#    if not EVAL:  # 
#        EVAL = 'unknown'
#        # return 
#    main_json = './track/sub/%s/main.json'%EVAL 
#    if os.path.exists(main_json):
#        de = json.load(open(main_json, 'r'))
#        de.append(d)
#    else:
#        de = [d]
#    json.dump(de, open(main_json, 'w'), indent=2)
#    logger.info('write to %s'%main_json)
#def print_ifnot(arg, default, custom_str):
#    """ Get config str for arg, ignoring if set to default. """
#    not_default = arg != default
#    return custom_str if not_default else '' 
#
def format_e(n):
    a = '%E'%n
    return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E-0')[1]
#def hash_file(file_name):
#    # BUF_SIZE is totally arbitrary, change for your app!
#    BUF_SIZE = 65536  # lets read stuff in 64kb chunks!
#    md5 = hashlib.md5()
#    # sha1 = hashlib.sha1()
#    with open(file_name, 'rb') as f:
#        while True: 
#            data = f.read(BUF_SIZE)
#            if not data: break
#            md5.update(data)
#            # sha1.update(data) 
#    # print("MD5: {0}".format(md5.hexdigest()))
#    # print("SHA1: {0}".format(sha1.hexdigest()))
#    # print("MD5: {0}".format(hash_str))
#    hash_str = md5.hexdigest()[:6]
#    hashed_file = json.load(open('.results/hash.json', 'r')) 
#    if hash_str in hashed_file:
#        assert(os.path.abspath(hashed_file[hash_str]) == os.path.abspath(file_name)), f'hash: {hash_str} has two value: {hashed_file[hash_str]} & {file_name}'
#    hashed_file[hash_str] = file_name 
#    json.dump(hashed_file, open('.results/hash.json', 'w'), indent=4) 
#    return hash_str
#
#def write_snapshot2slurmdir(save_model_fun, exp_dir):
#    if not os.path.exists(f"/checkpoint/xiaohui/{os.getenv('SLURM_JOB_ID', None)}"): 
#        return None 
#        # and ( (time.time()-t1) / 60 > 5 or nwrite == 0):
#    slurm_model_p = f"/checkpoint/xiaohui/{os.getenv('SLURM_JOB_ID', None)}/snapshot.pth"
#    savedp = save_model_fun(slurm_model_p) 
#    linkedp = os.path.join(exp_dir, 'snapshot.pth') 
#    logger.info('[Write] to slurm dir: {}', slurm_model_p) 
#    if os.path.exists(linkedp): 
#        if not os.path.islink(linkedp): 
#            # exists snapshot.pth file in target dir; rename it s.t. we can do soft-link 
#            os.rename(linkedp, linkedp+'.bak')
#        else: # its a softlink itself 
#            os.unlink(linkedp)
#    os.symlink(savedp, linkedp) # create softlink 
#def write_resumebash2slurmdir(full_exp_dir):
#    # ---------- write to the slurm dir with resume.sh ----------- #
#    jid = os.getenv('SLURM_JOB_ID', None) 
#    EVAL = os.getenv('EVAL', None) 
#    logger.debug(f'JOB ID: {jid}')
#    slurm_dir = f'/checkpoint/xiaohui/{jid}'          
#    if os.path.exists(slurm_dir) and not os.path.exists(slurm_dir + '/resume.sh'):
#        logger.info('-- write to resume.sh in %s'%slurm_dir)
#        with open(os.path.join(slurm_dir, 'resume.sh'), 'w') as f:
#            if EVAL: f.write('export EVAL=%s \n '%EVAL)
#            f.write('python '+sys.argv[0]\
#                +' --resume %s/snapshot.pth \n'%full_exp_dir) #os.path.join(exp_dir, cfg.exp_name))
#def clean_resumebash2slurmdir():
#    # ---------- write to the slurm dir with resume.sh ----------- #
#    jid = os.getenv('SLURM_JOB_ID', None) 
#    EVAL = os.getenv('EVAL', None) 
#    logger.debug(f'JOB ID: {jid}')
#    slurm_dir = f'/checkpoint/xiaohui/{jid}'
#    if os.path.exists(slurm_dir+'/resume.sh'):
#        lines = open(slurm_dir+'/resume.sh', 'r').readlines()
#
#        with open(slurm_dir+'/resume.sh', 'w') as f:
#            f.write('') # write empty thing to it, overwrite previous resume 
#        logger.info('-- write to resume_1.sh in %s'%slurm_dir)
#        with open(os.path.join(slurm_dir, 'resume_1.sh'), 'w') as f:
#            for l in lines: 
#                f.write(l)
#
#
#def append2jsondict(json_path, key, value):
#    ## json_path = os.path.dirname(args.path[0]) + '/eval_fid.json' 
#    if os.path.exists(json_path):
#        fid_score_dict = json.load(open(json_path, 'r'))
#    else:
#        fid_score_dict = {}
#    fid_score_dict[key] = value 
#    ## fid_score_dict[datestr] = {'gt': gt, 'pred': pred, 'fid':fid_value, 'url': experiment.url}
#    json.dump(fid_score_dict, open(json_path, 'w'), indent=4)
#    return json_path 
#
#if __name__ == '__main__':
#    import sys 
#    file_name = sys.argv[1] 
#    hash_file(file_name)
