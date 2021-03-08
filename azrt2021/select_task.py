"""
select_task.py
select task based on task_id input;
"""
def select_task(task_id, task_csv_txt):
    """
    based on task_id, and return csv_info, ext;
    "0=norm vs. demented;\n1=nondemented vs. demented;\n2=norm vs. mci;\n"+\
                "3=mci vs. demented;"
    """
    task_id = int(task_id)
    csv_info = None
    with open(task_csv_txt, 'r') as infile:
        for idx, line in enumerate(infile):
            if idx == task_id:
                csv_info, ext = line.split(',')
                ext = ext.strip()
                break
    return csv_info, ext
