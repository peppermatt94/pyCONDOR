import json
import argparse
import os
import time
from pathlib import Path
from DTAtools.settings import CODE_PATHS
import colorama
from colorama import init, Fore, Back, Style
import sys
from tree import walk_directory
from rich import print
from rich.filesize import decimal
from rich.markup import escape
from rich.text import Text
from rich.tree import Tree
import pathlib


def print_dir_content(code_paths):
    try:
        directory = os.path.abspath(code_paths)
    except IndexError:
        print("[b]Usage:[/] python tree.py <DIRECTORY>")
    else:
        tree = Tree(
            f":open_file_folder: [link file://{directory}]{directory}",
            guide_style="bold bright_blue",
        )
        walk_directory(pathlib.Path(directory), tree)
        print(tree)

def main():
    init()
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", "-f", help="select the file to whic add a commit")
    parser.add_argument("--commit", "-c", help="commit to add")
    parser.add_argument("--info", "-i", help="Some additive information")
    parser.add_argument("--showdir", "-s", help = "show a tree of the code directory",  action='store_true')
    parser.add_argument("--cancealfile", "-rm", help="canceal a file from the site view")
    args = parser.parse_args()
    
    commit_path = (CODE_PATHS /  "commit.json").as_posix()
    if Path(commit_path).exists():
        with open(commit_path, "r") as f:
            data = json.load(f)
    else:
        data = {
            'Files' : [
        ]
        }
        

    if args.filename != None:
        file_path= (CODE_PATHS /  args.filename).as_posix()
        if  Path(file_path).exists():
            new_file = {"name" :args.filename }
            if args.commit != None:
                new_commit = {"commit": args.commit}
            else:
                new_commit = {"commit": None}
            if args.info != None:
                new_info ={"info": args.info}
            else:
                new_info = {"info": None}
            last_mod = time.ctime(os.path.getmtime((CODE_PATHS / args.filename).as_posix()))
            modified = {"last_modified": last_mod}
            new_file.update(new_commit)
            new_file.update(modified)
            new_file.update(new_info)
            name_list = [file["name"] for file in data["Files"]]
            if args.filename in name_list:
                index = name_list.index(args.filename)
                data["Files"].pop(index)
            data['Files'].append(new_file)
            with open(commit_path , "w") as f:
                json.dump(data, f, indent = 5)
        else:
           sys.stdout.write(Fore.BLUE +Back.RED +"""
#############################
#   THE FILENAME PASSED     # 
#    IS NOT IN THE SOURCE   # 
#    CODES DIRECTORY.       #
# commit.py --showdir to see#
#       the dir content     #
#############################""")
           print_dir_content(CODE_PATHS.as_posix())
           exit(1)        
    if args.showdir:
        print_dir_content(CODE_PATHS.as_posix())
    
    if args.cancealfile != None: 
        name_list = [file["name"] for file in data["Files"]]
        if args.cancealfile in name_list:
            index = name_list.index(args.cancealfile)
            data["Files"].pop(index)
            with open(commit_path , "w") as f:
                json.dump(data, f, indent = 5)
        else:
            sys.stdout.write(Fore.BLUE +Back.RED +"""
#############################
#   THE FILENAME PASSED     # 
#    IS NOT IN THE SOURCE   # 
#    COMMITTED FILES.       #
#    committed files in     #
#       commit.json         #
#############################""")
            exit(1)
if __name__=='__main__':
    main()
