from __future__ import print_function

import base64
import getopt
import itertools
import json
import math
import netrc
import os.path
import ssl
import sys
import time
from getpass import getpass
import shutil

import credentials
from bs4 import BeautifulSoup
import requests


try:
    from urllib.parse import urlparse
    from urllib.request import urlopen, Request, build_opener, HTTPCookieProcessor
    from urllib.error import HTTPError, URLError
except ImportError:
    from urlparse import urlparse
    from urllib2 import urlopen, Request, HTTPError, URLError, build_opener, HTTPCookieProcessor

out_dir = "./Data/data/"
data_url = "https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0046_weekly_snow_seaice/data/"


start_year = 1979
end_year = 2020

'''
    Fairly inefficient, but it does work
'''


###### LOGIN AUTHENTICATION #####
def get_username():
    username = ''

    # new
    if credentials.username:
        return credentials.username

    # For Python 2/3 compatibility:
    try:
        do_input = raw_input  # noqa
    except NameError:
        do_input = input

    username = do_input('Earthdata username (or press Return to use a bearer token): ')
    return username


def get_password():
    password = ''

    # new
    if credentials.password:
        return credentials.password

    while not password:
        password = getpass('password: ')
    return password


def get_token():
    token = ''
    while not token:
        token = getpass('bearer token: ')
    return token


def get_login_credentials():
    """Get user credentials from .netrc or prompt for input."""
    credentials = None
    token = None

    try:
        info = netrc.netrc()
        username, account, password = info.authenticators(urlparse(URS_URL).hostname)
        if username == 'token':
            token = password
        else:
            credentials = '{0}:{1}'.format(username, password)
            credentials = base64.b64encode(credentials.encode('ascii')).decode('ascii')
    except Exception:
        username = None
        password = None

    if not username:
        username = get_username()
        if len(username):
            password = get_password()
            credentials = '{0}:{1}'.format(username, password)
            credentials = base64.b64encode(credentials.encode('ascii')).decode('ascii')
        else:
            token = get_token()

    return credentials, token



def get_login_response(url, credentials, token):
    opener = build_opener(HTTPCookieProcessor())

    req = Request(url)
    if token:
        req.add_header('Authorization', 'Bearer {0}'.format(token))
    elif credentials:
        try:
            response = opener.open(req)
            # We have a redirect URL - try again with authorization.
            url = response.url
        except HTTPError:
            # No redirect - just try again with authorization.
            pass
        except Exception as e:
            print('Error{0}: {1}'.format(type(e), str(e)))
            sys.exit(1)

        req = Request(url)
        req.add_header('Authorization', 'Basic {0}'.format(credentials))

    try:
        response = opener.open(req)
    except HTTPError as e:
        err = 'HTTP error {0}, {1}'.format(e.code, e.reason)
        if 'Unauthorized' in e.reason:
            if token:
                err += ': Check your bearer token'
            else:
                err += ': Check your username and password'
        print(err)
        sys.exit(1)
    except Exception as e:
        print('Error{0}: {1}'.format(type(e), str(e)))
        sys.exit(1)

    return response

#############################

def get_speed(time_elapsed, chunk_size):
    if time_elapsed <= 0:
        return ''
    speed = chunk_size / time_elapsed
    if speed <= 0:
        speed = 1
    size_name = ('', 'k', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')
    i = int(math.floor(math.log(speed, 1000)))
    p = math.pow(1000, i)
    return '{0:.1f}{1}B/s'.format(speed / p, size_name[i])


def output_progress(count, total, status='', bar_len=60):
    if total <= 0:
        return
    fraction = min(max(count / float(total), 0), 1)
    filled_len = int(round(bar_len * fraction))
    percents = int(round(100.0 * fraction))
    bar = '=' * filled_len + ' ' * (bar_len - filled_len)
    fmt = '  [{0}] {1:3d}%  {2}   '.format(bar, percents, status)
    print('\b' * (len(fmt) + 4), end='')  # clears the line
    sys.stdout.write(fmt)
    sys.stdout.flush()


def cmr_read_in_chunks(file_object, chunk_size=1024 * 1024):
    """Read a file in chunks using a generator. Default chunk size: 1Mb."""
    while True:
        data = file_object.read(chunk_size)
        if not data:
            break
        yield data

def get_folder(url, end_year):
    # get the proper out file path for organization (only look at start year/month)
    sy = int(url[-25:-21]) # start month
    sm = int(url[-21:-19]) # start month
    ey = int(url[-16:-12]) # end month
    em = int(url[-12:-10]) # end month
    month = sm
    year = sy

    ## checking the end date to make sure file is placed in right month - if there
    ## are more 4 or more days belonging to that month, thats where the file goes
    ed = int(url[-10:-8]) # ending day

    if ed>=4:
        month = em
        year = ey

    # if that falls outside of the proper range then skip it by returning -1 for the year
    if year > end_year:
        year = -1

    return year, month




# Downloading
def cmr_download(urls, end_year, force=False, quiet=False, path=out_dir):
    """Download files from list of urls."""
    if not urls:
        return

    url_count = len(urls)
    if not quiet:
        print('Downloading {0} files...'.format(url_count))
    credentials = None
    token = None

    for index, url in enumerate(urls, start=1):
        if not credentials and not token:
            p = urlparse(url)
            if p.scheme == 'https':
                credentials, token = get_login_credentials()

        filename = url.split('/')[-1]
        if not quiet:
            print('{0}/{1}: {2}'.format(str(index).zfill(len(str(url_count))),
                                        url_count, filename))

        try:
            response = get_login_response(url, credentials, token)
            length = int(response.headers['content-length'])
            try:
                if not force and length == os.path.getsize(filename):
                    if not quiet:
                        print('  File exists, skipping')
                    continue
            except OSError:
                pass
            count = 0
            chunk_size = min(max(length, 1), 1024 * 1024)
            max_chunks = int(math.ceil(length / chunk_size))
            time_initial = time.time()

            # for organization of files
            year, mnth = get_folder(url, end_year)
            if year==-1:
                continue

            with open(path + str(year) + '/' + str(mnth) + '/' + filename, 'wb') as out_file:
                for data in cmr_read_in_chunks(response, chunk_size=chunk_size):
                    out_file.write(data)
                    if not quiet:
                        count = count + 1
                        time_elapsed = time.time() - time_initial
                        download_speed = get_speed(time_elapsed, count * chunk_size)
                        output_progress(count, max_chunks, status=download_speed)



            if not quiet:
                print()
        except HTTPError as e:
            print('HTTP error {0}, {1}'.format(e.code, e.reason))
        except URLError as e:
            print('URL error: {0}'.format(e.reason))
        except IOError:
            raise



##################################################################



def get_urls(url=data_url, ext='.bin'):
    '''
        Get the urls of all files to download
    '''
    page = requests.get(url).text
    soup = BeautifulSoup(page, 'html.parser')
    url_list = [url + '/' + node.get('href') for node in soup.find_all('a') if node.get('href').endswith(ext)]


    # Remove duplicates
    url_list = url_list[::2]

    # printing
    # for u in url_list:
    #     print(u)
    # print("Total: "+str(len(url_list)))


    return url_list

def setup_dirs(data_path, year):
    '''
        Given a year, set up the directory and all subdirectories for it
    '''

    year_path = str(year)
    path = data_path + year_path

    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
        return


    for i in range(1, 13):
        m_path = path + '/' + str(i)
        try:
            os.mkdir(m_path)
        except OSError:
            print ("Creation of the directory %s failed" % m_path)

def setup(start, end, clean=False):

    # if clean is true, delete the current sub dirs and start from scratch
    try:
        shutil.rmtree(out_dir) # removes data folder
        os.mkdir(out_dir) # creates empty data folder
    except OSError as e:
        print ("Could not delete existing folders." % (e.filename, e.strerror))

    for i in range(start, end+1):
        setup_dirs(out_dir, i)


def filter_urls(urls, starty, endy):
    good_urls = []
    for u in urls:
        # getting start and end year in url
        sy = int(u[-25:-21])
        ey = int(u[-16:-12])

        # checking if within range
        if (sy >= starty) and (ey <= endy):
            good_urls.append(u)

    return good_urls




#################################################
if __name__=="__main__":


    setup(start_year, end_year, clean=True) # only need to run once
    urls = get_urls()
    url_list = filter_urls(urls, start_year, end_year)

    for u in url_list:
        print(u)


    try:
        cmr_download(url_list, end_year, force=False, quiet=False)
    except KeyboardInterrupt:
        quit()
