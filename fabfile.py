from distutils.util import strtobool

from fabric.api import local, abort, run, sudo
from fabric.context_managers import cd, settings, hide, shell_env
from fabric.contrib.console import confirm
from getpass import getpass
from fabric.utils import puts

from fabric.state import env
env.control_dir = 'pipecontrol'

def with_sudo():
    """
    Prompts and sets the sudo password for all following commands.

    Use like

    fab with_sudo command
    """
    env.sudo_password = getpass('Please enter sudo password: ')
    env.password = env.sudo_password

def down():
    with cd(env.control_dir):
        sudo('docker-compose down')


def get_branch(gitdir):
    """
    Gets the branch of a git directory.

    Args:
        gitdir: path of the git directory

    Returns: current active branch

    """
    return local('git symbolic-ref --short HEAD', capture=True)

def pull():
    with cd(env.control_dir):
        branch = get_branch(env.control_dir)
        run('git reset --hard')
        run('git clean -fd')
        run('git checkout {}'.format(branch))
        run('git pull origin ' + branch)


def build():
    with cd(env.control_dir):
        sudo('docker-compose build pipecontrol')

def start():
    with cd(env.control_dir):
        sudo('docker-compose up -d pipecontrol')

def sync_files():
    local('scp dj_local_conf.json ' + env.host_string + ':' + env.control_dir)


def deploy():
    with settings(warn_only=True):
        down()
        pull()
        sync_files()
        build()
        start()