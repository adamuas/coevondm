#!/usr/bin/env python
# encoding: utf-8
#!/usr/bin/python
import smtplib
from datetime import datetime
 
 
def noticeEMail(name = ''):
    """
    Sends an email message through GMail once the script is completed.  
    Developed to be used with AWS so that instances can be terminated 
    once a long job is done. Only works for those with GMail accounts.
    
    starttime : a datetime() object for when to start run time clock

    usr : the GMail username, as a string

    psw : the GMail password, as a string 
    
    fromaddr : the email address the message will be from, as a string
    
    toaddr : a email address, or a list of addresses, to send the 
             message to
    """
 

    
    # Initialize SMTP server
    msg = dict();
    msg['Subject'] = 'Your job is Complete {0}'.format(name);
    msg['From'] = 'research.abdullah@gmail.com'
    msg['To'] = 'khyx1asa@nottingham.edu.my'

    # Send the message via our own SMTP server, but don't include the
    # envelope header.
    s = smtplib.SMTP('localhost')
    s.sendmail(me, [you], msg.as_string())

#
#noticeEMail();
# print "done";
