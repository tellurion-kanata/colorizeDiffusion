import os
import json
import threading

from glob import glob


posts = []

metaroot = "/raid/metadata"
save_path = "/raid/danbooru/tags"
num_threads = 8


def save_json(path, img_dict):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    filename = os.path.join(path, str(img_dict['id']) + '.json')
    with open(filename, 'w') as file:
        json.dump(img_dict, file, indent=1)


def processing(thread_id, post_files):
    for post_file in post_files:
        with open(post_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                img_dict = {}
                post = json.loads(line)

                try:
                    id = int(post['id'])
                except:
                    continue
                tail = id % 1000
                tag_string = post['tag_string'].replace(' ', ',')
                tag_split = post['tag_string'].split(" ")

                img_dict['id'] = id
                img_dict['tag_string'] = tag_string
                img_dict['tag_split'] = tag_split

                path = os.path.join(save_path, f'{tail:04d}')
                save_json(path, img_dict)


def create_threads():
    files = glob(os.path.join(metaroot, "post*.json"))
    data_size = len(files)
    thread_size = data_size // num_threads
    threads = []
    for t in range(num_threads):
        if t == num_threads - 1:
            thread = threading.Thread(target=processing, args=(t, files[t*thread_size: ]))
        else:
            thread = threading.Thread(target=processing, args=(t, files[t*thread_size: (t+1)*thread_size]))
        threads.append(thread)
    for t in threads:
        t.start()
    thread.join()

if __name__ == '__main__':
    create_threads()