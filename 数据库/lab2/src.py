from flask import Flask, request, session, jsonify
from flask import render_template
from flask import redirect, url_for
from flask_sqlalchemy import SQLAlchemy
import pymysql
import os
from werkzeug.utils import secure_filename

app = Flask(__name__, template_folder='templates')
app.secret_key = 'syx21071472'
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '142857',
    'db': 'db2024',
    'cursorclass': pymysql.cursors.DictCursor
}
# 创建数据库连接
connection = pymysql.connect(**db_config)


# 前端部分
@app.route('/', methods=['GET', 'POST'])
def root():
    return render_template('index.html')


@app.route('/login', methods=['POST'])
def login():
    userid = request.form.get('id')
    password = request.form.get('password')
    identity = request.form.get('identity')
    try:
        with connection.cursor() as cursor:
            # 查询数据库，判断学号和密码是否匹配
            if identity == '学生':
                query = "SELECT * FROM student WHERE sid = %s AND spassword = %s"
            else:
                query = "SELECT * FROM administrator WHERE aid = %s AND apassword = %s"
            cursor.execute(query, (userid, password))
            result = cursor.fetchone()
            if result:
                if identity == '学生':
                    return redirect(url_for('student_management'))
                else:
                    return redirect(url_for('administrator_management'))
            else:
                return '登录失败！用户名或密码错误'
    except Exception as e:
        # 处理异常
        return '登录失败：' + str(e)


@app.route('/register', methods=['POST'])
def register():
    username = request.form.get('name')
    password = request.form.get('password')
    identity = request.form.get('identity')
    userid = request.form.get('id')
    try:
        with connection.cursor() as cursor:
            # 查询数据库，判断学号和密码是否匹配
            if (identity == '学生'):
                query = "SELECT count(*) FROM student WHERE sid = %s"
                insert_query = "INSERT INTO student (sid, sname, spassword) VALUES (%s, %s, %s)"
            else:
                query = "SELECT count(*) FROM administrator WHERE aid = %s"
                insert_query = "INSERT INTO administrator (aid, aname, apassword) VALUES (%s, %s, %s)"
            cursor.execute(query, (userid,))
            result = cursor.fetchone()
            if result is not None and result['count(*)'] > 0:
                return '该用户已经存在'
            else:
                cursor.execute(insert_query, (userid, username, password))
                connection.commit()
                if identity == '学生':
                    return redirect(url_for('student_management', userid=userid))
                else:
                    return redirect(url_for('administrator_management'))
    except Exception as e:
        # 处理异常
        return '注册失败：' + str(e)


@app.route('/student_management')
def student_management():
    userid = request.args.get('userid')
    try:
        with connection.cursor() as cursor:
            query = "SELECT sid, sname, snumber, room_number,sfigure FROM student"
            cursor.execute(query)
            students = cursor.fetchall()
    except Exception as e:
        return '获取学生信息失败：' + str(e)

    return render_template('student.html', userid=userid, student=students[0])


@app.route('/smodify_information', methods=['POST'])
def smodify_information():
    userid = request.form['userid']
    password = request.form['password']
    contact = request.form['contact']
    room_number = request.form['room-number']
    try:
        avatar = request.files['avatar']
        avatar_filename = secure_filename(avatar.filename)
        # avatar_path = os.path.join('./figure', avatar_filename)
        avatar_path = "./static/figure/" + avatar_filename
        avatar.save(avatar_path)
    except Exception as e:
        return '保存头像失败：' + str(e)
    try:
        with connection.cursor() as cursor:
            ret = 0
            cursor.callproc("ChangeRoom", [userid, room_number, ret])  # 进行房间更新
            if ret == 4:
                raise Exception('房间已满')
            query = "UPDATE student SET spassword=%s, snumber=%s, room_number=%s, sfigure=%s WHERE sid=%s"
            cursor.execute(query, (password, contact, room_number, avatar_filename, userid))
            connection.commit()
    except Exception as e:
        return '修改信息失败：' + str(e)

    return redirect('/student_management?userid=' + userid)


@app.route('/report_repair', methods=['POST'])
def report_repair():
    userid = request.form['userid']
    problem_description = request.form['problem-description']
    try:
        with connection.cursor() as cursor:
            query = "SELECT room_number FROM student WHERE sid=%s"
            cursor.execute(query, userid)
            rooms = cursor.fetchall()
            room_number = rooms[0]['room_number']
            assert room_number
    except Exception as e:
        return '该用户未绑定房间号,' + str(e)
    try:
        with connection.cursor() as cursor:
            query = "INSERT INTO maintenance (mroom, mmessage, mstatus) VALUES (%s, %s, %s)"
            cursor.execute(query, (room_number, problem_description, 0))
            connection.commit()
    except Exception as e:
        return '报修失败：' + str(e)
    return redirect('/student_management?userid=' + userid)


@app.route('/administrator_management')
def administrator_management():
    userid = request.args.get('userid')
    try:
        with connection.cursor() as cursor:
            query = "SELECT aid, aname, anumber, apartment ,afigure FROM administrator"
            cursor.execute(query)
            administrators = cursor.fetchall()
    except Exception as e:
        return '获取学生信息失败：' + str(e)

    return render_template('administrator.html', userid=userid, administrator=administrators[0])


@app.route('/amodify_information', methods=['POST'])
def amodify_information():
    userid = request.form['userid']
    password = request.form['password']
    contact = request.form['contact']
    apartment = request.form['apartment-number']
    try:
        avatar = request.files['avatar']
        avatar_filename = secure_filename(avatar.filename)
        # avatar_path = os.path.join('./figure', avatar_filename)
        avatar_path = "./static/figure/" + avatar_filename
        avatar.save(avatar_path)
    except Exception as e:
        return '保存头像失败：' + str(e)
    try:
        with connection.cursor() as cursor:
            query = "UPDATE administrator SET apassword=%s, anumber=%s, apartment=%s, afigure=%s WHERE aid=%s"
            cursor.execute(query, (password, contact, apartment, avatar_filename, userid))
            connection.commit()
    except Exception as e:
        return '修改信息失败：' + str(e)

    return redirect('/administrator_management?userid=' + userid)


@app.route('/deal_repair', methods=['POST'])
def deal_repair():
    query_all = "SELECT * FROM maintenance"  # 查询所有报修记录
    query_unsolved = "SELECT * FROM maintenance WHERE mstatus = 0"  # 查询未解决报修记录
    option = request.form.get('repair-option')
    try:
        with connection.cursor() as cursor:
            cursor.execute(query_all)
            all_repairs = cursor.fetchall()
            cursor.execute(query_unsolved)
            unsolved_repairs = cursor.fetchall()
    except Exception as e:
        return '查看报修失败：' + str(e)
    return render_template('check_repair.html', all_repairs=all_repairs, unsolved_repairs=unsolved_repairs,
                           option=option)


@app.route('/submit_repairs', methods=['POST'])
def submit_repairs():
    selected_repairs = request.form.getlist('selected_repairs')

    try:
        with connection.cursor() as cursor:
            # 将选中的报修记录状态改为已解决（状态码为1）
            for repair_id in selected_repairs:
                query = "UPDATE maintenance SET mstatus = 1 WHERE mid = %s"
                cursor.execute(query, (repair_id,))
            connection.commit()
    except Exception as e:
        return '提交失败：' + str(e)

    return redirect('/administrator_management')


@app.route('/visitor_checkin', methods=['POST'])
def visitor_checkin():
    visitor_name = request.form['visitor-name']
    arrive_time = request.form['arrive_time']
    leave_time = request.form['leave_time']
    try:
        with connection.cursor() as cursor:
            query = "INSERT INTO visitor (vname, vdata, ldata) VALUES (%s, %s, %s)"
            cursor.execute(query, (visitor_name, arrive_time, leave_time))
            connection.commit()
    except Exception as e:
        return '访客登记失败：' + str(e)
    return redirect('/administrator_management')


# 新增路由用于获取所有访客信息
@app.route('/get_all_visitors', methods=['GET'])
def get_all_visitors():
    try:
        with connection.cursor() as cursor:
            query = "SELECT * FROM visitor"
            cursor.execute(query)
            visitors = cursor.fetchall()
            # 将查询到的访客信息转换为字典列表
            return render_template('visitors.html', visitors=visitors)
    except Exception as e:
        return str(e)


@app.route('/student_info', methods=['GET', 'POST'])
def student_info():
    option = request.form.get('check-option')
    if option == 'stu':
        try:
            with connection.cursor() as cursor:
                query = "SELECT * FROM student where room_number is not null"
                cursor.execute(query)
                students = cursor.fetchall()
                return render_template('student_management.html', students=students)
        except Exception as e:
            return '学生信息加载失败: ' + str(e)
    else:
        # 可以做成根据学生房间号第一个数字来判断是哪个楼栋
        try:
            with connection.cursor() as cursor:
                query = "SELECT * FROM room"
                cursor.execute(query)
                room = cursor.fetchall()
                cursor.execute("SELECT calculate_avg_students_per_room() AS average")
                result = cursor.fetchone()
                avg_students_per_room = result['average']
                return render_template('room_management.html', rooms=room, avg_students_per_room=avg_students_per_room)
        except Exception as e:
            return '房间信息加载失败: ' + str(e)


@app.route('/del_stu', methods=['POST'])
def del_stu():
    try:
        with connection.cursor() as cursor:
            delete_students = request.form.getlist('delete_student')
            if delete_students:
                # 根据传入的学生学号列表获取对应学生的房间号
                query = "SELECT room_number FROM student WHERE sid IN (%s)"
                placeholders = ', '.join(['%s'] * len(delete_students))

                # 更新房间信息
                for student in delete_students:
                    ret = 0
                    cursor.callproc("ChangeRoom", [student, 0, ret])  # 进行房间更新
                # 将房间号为delete_students中的学生的房间号置为空
                update_query = "UPDATE student SET room_number = NULL WHERE sid IN (%s)"
                update_query = update_query % placeholders
                cursor.execute(update_query, tuple(delete_students))

                connection.commit()
        return redirect('/administrator_management')
    except Exception as e:
        return '删除学生失败: ' + str(e)


@app.route('/logout', methods=['POST'])
def logout():
    session.pop('user_id', None)

    # 重定向到登录页面
    # return redirect(url_for('login'))
    return render_template('index.html')


if __name__ == '__main__':
    # with app.app_context():

    app.run(debug=True)
