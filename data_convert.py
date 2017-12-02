import re

from nltk.tokenize import sent_tokenize
from pymysql import *

import proto_files.compiled.Email_pb2 as EmailPb2

conn = connect(
    host='hk1.lee-service.com',
    user='mathteam',
    password='HITmathTEAM',
    db='enron',
    charset='utf8mb4',
    cursorclass=cursors.DictCursor
)

dataset_size = 255636

# dataset_size = 25

batch_size = 500


# batch_size = 10


def contains_alpha(s):
    return re.match(r'.*[a-zA-Z0-9]+', s)


with conn.cursor() as cursor:
    steps = dataset_size // batch_size + 1
    with open('data/enron_emails.pb', 'wb') as f:
        for i in range(steps):
            if (i * batch_size > dataset_size):
                break
            cursor.execute('''
                SELECT messages.messageid,senderid as `from`, recipientid as `to`,messages.subject as `subject`,bodies.body
                FROM messages,bodies,recipients
                WHERE messages.messageid=bodies.messageid AND messages.messageid=recipients.messageid
                ORDER BY messages.messageid
                LIMIT %d,%d
                ''' % (i * batch_size, batch_size))
            for doc in cursor.fetchall():
                # print("|".center(200, '-'))
                if not doc['subject']:
                    doc['subject'] = 'NONE'
                # print("SUBJECT %s" % doc['subject'])
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
                # print(email)
                # f.writelines()
                compiled_data = email.SerializeToString()
                bin_size = email.ByteSize()
                d = (bin_size).to_bytes(4, byteorder='little', signed=False) + compiled_data
                # print(bin_size)
                # print(hex(bin_size))
                # print(d)
                f.write(d)
# print(docs)
