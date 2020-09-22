import smtplib

sender="gaire.avinash@gmail.com"
receiver="gaire.avinash@gmail.com"
passowrd="email2me"

message="hello how are you"
server=smtplib.SMTP('smtp.gmail.com',587)
server.starttls()
server.login(sender,passowrd)
print("hello")
server.sendmail(sender,receiver,message)
print("emailsent :",receiver)
