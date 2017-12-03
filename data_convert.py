import re

from nltk.tokenize import sent_tokenize
from pymysql import *

import proto_files.compiled.Email_pb2 as EmailPb2


def contains_alpha(s):
    return re.match(r'.*[a-zA-Z0-9]+', s)


def pack_data(email):
    return (email.ByteSize()).to_bytes(4, byteorder='little', signed=False) + email.SerializeToString()


def convert(path, batch_size, limit):
    conn = connect(
        host='hk1.lee-service.com',
        user='mathteam',
        password='HITmathTEAM',
        db='enron',
        charset='utf8mb4',
        cursorclass=cursors.DictCursor
    )

    main_sql = '''
        SELECT messages.messageid,senderid as `sender`, recipientid as `recipient`,messages.subject as `subject`,bodies.body
        FROM messages,bodies,recipients
        WHERE messages.messageid=bodies.messageid AND messages.messageid=recipients.messageid
        ORDER BY messages.messageid
        '''

    # dataset_size = 25

    with conn.cursor() as cursor:
        cursor.execute('SELECT COUNT(*) AS dataset_size FROM (%s) AS T' % main_sql)
        dataset_size = 0
        for d in cursor.fetchall():
            dataset_size = d['dataset_size']
        # print(dataset_size)
        steps = dataset_size // batch_size + 1

        doc_count = 0

        with open(path, 'wb') as f:
            for i in range(steps):
                if limit != 0 and doc_count >= limit:
                    break
                if i * batch_size > dataset_size:
                    break
                query = main_sql + ('LIMIT %d,%d' % (i * batch_size, batch_size))
                # print(query)
                cursor.execute(query)
                for doc in cursor.fetchall():
                    # print("|".center(200, '-'))
                    if not doc['subject']:
                        doc['subject'] = 'NONE'
                    body = []
                    for line in doc['body'].splitlines():
                        if line.find('---') != -1:
                            break
                        body.append(line)
                    body = ''.join(body)
                    # print(body)
                    sent_tks = [s for s in sent_tokenize(body) if contains_alpha(s)]
                    # print(sent_tks)
                    email = EmailPb2.Email()
                    doc['body'] = body
                    for k in doc.keys():
                        if k == 'body':
                            email.body.extend(sent_tks)
                        else:
                            setattr(email, k, doc[k])
                    f.write(pack_data(email))
                    doc_count += 1


if __name__ == '__main__':
    convert('data/enron_emails.pb', 1000, 0)
