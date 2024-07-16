-- 0为女1为男
create table apartment(
aid int,
gender bool, 
primary key (aid)
);

create table student(
sid char(10),
sname varchar(12),   -- 一个汉字3个字节
snumber char(11),  -- 联系方式
spassword varchar(15), -- 最长15位的密码
sfigure varchar(30) default '0.jpg', -- 存储图片路径
room_number INT,
primary key (sid)
);

create table room(
rnumber int,
rstatus int default 0,  -- 人数
student1_id char(10),
student2_id char(10),
student3_id char(10),
student4_id char(10),
FOREIGN KEY (student1_id) REFERENCES student (sid),
FOREIGN KEY (student2_id) REFERENCES student (sid),
FOREIGN KEY (student3_id) REFERENCES student (sid),
FOREIGN KEY (student4_id) REFERENCES student (sid),
primary key (rnumber)
);

create table administrator(
apartment int,
aid char (10),
aname varchar(12),
anumber char(11),  -- 联系方式
afigure varchar(30) default '0.jpg',
apassword varchar(15),
primary key (aid)
);

create table visitor(
vname varchar(12),
vdata datetime,  -- 来访日期
ldata datetime,  -- 离开日期
primary key(vname)
);
-- 修改使得在报修得到解决后还可以重新报修，增加id
create table maintenance(
mid int,
mroom int,
mstatus int,
mmessage varchar(300),
primary key(mid),
foreign key (mroom) references room (rnumber)
);