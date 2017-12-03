import proto_files.compiled.Email_pb2 as EmailPb2

from collections import defaultdict


def pack_date(email):
    return (email.ByteSize()).to_bytes(4, byteorder='little', signed=False) + email.SerializeToString()


def readobj(path, limit):
    with open(path, 'rb') as f:
        ct = 0
        bs = f.read(4)
        while bs:
            if limit != 0 and ct >= limit:
                break
            size = int.from_bytes(bs, 'little', signed=False)
            em = EmailPb2.Email()
            em.ParseFromString(f.read(size))
            yield em
            ct += 1
            bs = f.read(4)


def construct_no_dup(src, dst, limit):
    ct = 0
    with open(dst, 'wb') as f:
        seen = set()
        for em in readobj(src, limit):
            # print(em)
            triple = int(em.messageid), int(em.sender), int(em.recipient)
            if triple not in seen:
                # print(triple)
                seen.add(triple)
                f.write(pack_data(em))
                ct += 1
    return ct


def classify_by_sender(src, dst, limit, train_data_percent):
    ct = 0
    data = defaultdict(list)
    seen = set()
    for em in readobj(src, limit):
        # print(em)
        spec = int(em.messageid), int(em.sender)
        if spec not in seen:
            # print(triple)
            seen.add(spec)
            data[spec[1]].append(em)
            ct += 1
    for k, ems in data.items():
        with open(dst % ('train', k), 'wb') as ft:
            with open(dst % ('validate', k), 'wb') as fv:
                size = len(ems)
                div = int(size * train_data_percent)
                for i in range(size):
                    if size == 1 or i < div:
                        ft.write(pack_data(ems[i]))
                    else:
                        fv.write(pack_data(ems[i]))
    return ct


def construct_dataset(src, dst, limit, train_num):
    ct = 0
    seen = set()
    with open(dst % 'train', 'wb') as ft:
        with open(dst % 'validate', 'wb') as fv:
            for em in readobj(src, limit):
                spec = int(em.messageid), int(em.sender)
                if spec not in seen:
                    # print(triple)
                    seen.add(spec)
                    ct += 1
                    if ct <= train_num:
                        ft.write(pack_data(em))
                    else:
                        fv.write(pack_data(em))
    return ct


if __name__ == '__main__':
    print(construct_dataset('data/enron_emails.pb', 'data/%s.pb', 0, 240000))
