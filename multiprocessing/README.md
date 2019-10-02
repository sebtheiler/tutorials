# Multiprocessing for Data Scientists in Python
#### Why pay for a powerful CPU if you can’t use all of it?

Available here: https://medium.com/analytics-vidhya/multiprocessing-for-data-scientists-in-python-427b2ff93af1

If you ever encounter the error: `FileExistsError: [Errno 17] File exists: 'data'` run:
```
import SharedArray
SharedArray.delete('data')
