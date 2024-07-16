-- 学生修改房间后添加房间中的对应学号，并从原来的房间移除，检查房间的状态是否有空位
DELIMITER //
CREATE PROCEDURE ChangeRoom(
  IN stud_id CHAR(10),
  IN new_room_number INT,
  OUT ret INT
)
label :BEGIN
  DECLARE old_room_number INT;
  DECLARE old_room_status INT;
  DECLARE new_room_status INT;
  DECLARE empty_room_count INT;
  
  
  -- 获取当前学生的原始房间编号和房间状态
  SELECT room_number INTO old_room_number
  FROM student
  WHERE sid = stud_id;
  
  -- 如果转入房间与原房间相同，则直接结束存储过程
  IF old_room_number = new_room_number THEN
    SET ret = 0;
    LEAVE label;
  END IF;
  
  SELECT rstatus INTO old_room_status
  FROM room
  WHERE rnumber = old_room_number;
  
 
  
  -- 检查原始房间是否有其他学生
  IF old_room_status > 0 THEN
    -- 从原始房间移除学生
    UPDATE room
    SET
      student1_id = CASE WHEN student1_id = stud_id THEN NULL ELSE student1_id END,
      student2_id = CASE WHEN student2_id = stud_id THEN NULL ELSE student2_id END,
      student3_id = CASE WHEN student3_id = stud_id THEN NULL ELSE student3_id END,
      student4_id = CASE WHEN student4_id = stud_id THEN NULL ELSE student4_id END,
      rstatus = rstatus - 1
    WHERE rnumber = old_room_number;
  END IF;
  
 -- 判断是否是转走
  if new_room_number =0 then
	set ret = 5;
    leave label;
	end if;
  
  -- 更新学生的房间编号
  UPDATE student
  SET room_number = new_room_number
  WHERE sid = stud_id;
  
  -- 获取新房间的状态
  SELECT rstatus INTO new_room_status
  FROM room
  WHERE rnumber = new_room_number;
  
  -- 检查新房间是否已满
  IF new_room_status = 4 THEN
    set ret = 4;
    LEAVE label;
  END IF;
  
  -- 将学生添加到新房间
  UPDATE room
  SET
    student1_id = CASE WHEN student1_id IS NULL THEN stud_id ELSE student1_id END,
    student2_id = CASE WHEN student1_id != stud_id AND student2_id IS NULL THEN stud_id ELSE student2_id END,
    student3_id = CASE WHEN student1_id != stud_id AND student2_id != stud_id AND student3_id IS NULL THEN stud_id ELSE student3_id END,
    student4_id = CASE WHEN student1_id != stud_id AND student2_id != stud_id AND student3_id != stud_id AND student4_id IS NULL THEN stud_id ELSE student4_id END,
    rstatus = rstatus + 1
  WHERE rnumber = new_room_number;
  
END //
DELIMITER ;

-- 用于自动递增报修id
drop TRIGGER set_mid_trigger;
DELIMITER //
CREATE TRIGGER set_mid_trigger
BEFORE INSERT ON maintenance
FOR EACH ROW
BEGIN
    DECLARE last_mid INT;
    declare flag int;
    select count(*) from maintenance into flag;
    if flag > 0 then
		SET last_mid = (SELECT MAX(mid) FROM maintenance);
		SET NEW.mid = last_mid + 1;
	else 
		SET NEW.mid = 0;
	end if;
END //
DELIMITER ;

DELIMITER //
CREATE FUNCTION calculate_avg_students_per_room() RETURNS DECIMAL(10, 2)
Reads SQL data
BEGIN
    DECLARE total_students INT;
    DECLARE total_rooms INT;
    DECLARE avg_students DECIMAL(10, 2);

    SELECT COUNT(*) INTO total_students FROM student;

    SELECT COUNT(*) INTO total_rooms FROM room;

    -- 计算平均每个房间所住人数
    SET avg_students = total_students / total_rooms;

    RETURN avg_students;
END //

DELIMITER ;