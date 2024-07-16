-- 2.1
select  book.bid, book.bname, borrow.borrow_date from reader, book, borrow 
	where reader.rid = borrow.reader_ID and borrow.book_ID = book.bid and reader.rname='rose';
-- 2.2
select reader.rid, reader.rname from reader where reader.rid not in 
(select reader.rid from reader, borrow,reserve
	where reader.rid = borrow.reader_ID and reader.rid = reserve.reader_ID);
-- 2.3
select book.author from book, borrow 
	where book.bid = borrow.book_ID
		group by book.author
			order by count(book_ID) desc
				limit 1;
select book.author from book
	group by author
		order by sum(borrow_Times) desc
			limit 1;

-- 2.4
select b.bid, b.bname
from book b, borrow br 
where b.bid = br.book_id and b.bname like '%mysql%' and br.return_date is null;

-- 2.5
select r.rname
from reader r, borrow br where r.rid = br.reader_id
group by r.rid
having count(br.book_id) > 3;

-- 2.6
select r.rid, r.rname
from reader r
where not exists (
    select 1
    from borrow br
    join book b on br.book_id = b.bid
    where b.author = 'j.k. rowling' and br.reader_id = r.rid
);

-- 2.7
select r.rid, r.rname, count(br.book_id) as borrow_count
from reader r
join borrow br on r.rid = br.reader_id
where year(br.borrow_date) = 2024
group by r.rid
order by borrow_count desc
limit 3;

-- 2.8
create view readerborrowinfo as
select r.rid, r.rname, b.bid, b.bname, br.borrow_date
from reader r
join borrow br on r.rid = br.reader_id
join book b on br.book_id = b.bid;
SELECT 
    rbi.rid, COUNT(DISTINCT rbi.bid) AS distinct_books_borrowed
FROM
    readerborrowinfo rbi
WHERE
    YEAR(rbi.borrow_date) = 2024
GROUP BY rbi.rid;

-- 3.1
DELIMITER //

CREATE PROCEDURE updateReaderID(IN oldID CHAR(8), IN newID CHAR(8))
BEGIN
  DECLARE EXIT HANDLER FOR SQLEXCEPTION
  BEGIN
    SELECT "error";
    ROLLBACK;
  END;

  START TRANSACTION;

    alter table borrow drop foreign key borrow_ibfk_2;
    alter table reserve drop foreign key reserve_ibfk_2;
	UPDATE Reader SET rid = newID WHERE rid = oldID;
	UPDATE Borrow SET reader_ID = newID WHERE reader_ID = oldID;
	UPDATE Reserve SET reader_ID = newID WHERE reader_ID = oldID;
    alter table borrow add constraint borrow_ibfk_2 foreign key (reader_ID) references Reader(rid);
    alter table reserve add constraint reserve_ibfk_2 foreign key (reader_ID) references Reader(rid);
  COMMIT;
END //

DELIMITER ;

select rid from reader;

call updateReaderID('R006', 'R999');

select rid from reader;


-- 4
DELIMITER //
CREATE PROCEDURE borrowBook(IN input_reader_id CHAR(8), IN input_book_id CHAR(8), IN borrow_date DATE, OUT result VARCHAR(100))
label:BEGIN
    DECLARE book_count INT;
    declare flag int;
    DECLARE reservation_count INT;
    
    -- 同一天是否已经借阅过同一本图书
    SELECT COUNT(*) INTO book_count FROM borrow WHERE reader_ID = input_reader_id AND book_ID = input_book_id AND borrow_Date = borrow_date;
    IF book_count > 0 THEN
        set result = '借阅失败：同一天不允许重复借阅同一本图书。';
        leave label;
    END IF;
    
    -- 已经借阅的图书数量
    SELECT COUNT(*) INTO book_count FROM borrow WHERE reader_ID = input_reader_id AND return_Date IS NULL;
    IF book_count >= 3 THEN
        set result =  '借阅失败：该读者已经借阅了 3 本图书并且未归还。';
        leave label;
    END IF;
    
    -- 是否存在预约记录
    SELECT COUNT(*) INTO reservation_count FROM reserve WHERE reader_ID = input_reader_id AND book_ID = input_book_id;
    SELECT COUNT(*) INTO flag FROM reserve WHERE book_ID = input_book_id;
    IF reservation_count = 0 THEN
        set result = '借阅失败：该图书存在预约记录，您没有预约。';
        leave label;
    END IF;
    -- 删除借阅者对该图书的预约记录
    DELETE FROM reserve WHERE reader_ID = input_reader_id AND book_ID = input_book_id;
    -- 借阅成功
    UPDATE book SET borrow_Times = borrow_Times + 1, bstatus = 1 WHERE bid = input_book_id;
    INSERT INTO borrow (book_ID, reader_ID, borrow_Date) VALUES (input_book_id, input_reader_id, borrow_date);
    set result = '借阅成功。';
END //
DELIMITER ;

-- 4.1
set @debug = 0;
SET @result = "";
CALL borrowBook('R001', 'B008', '2024-05-09', @result);
SELECT @result;

SET @result = "";
CALL borrowBook('R001', 'B001', '2024-05-09', @result);
SELECT @result;

SET @result = "";
CALL borrowBook('R001', 'B001', '2024-05-09', @result);
SELECT @result;

SET @result = "";
CALL borrowBook('R005', 'B008', '2024-05-09', @result);
SELECT @result;

-- 5

DELIMITER //
CREATE PROCEDURE returnBook(IN input_reader_id CHAR(8), IN input_book_id CHAR(8), IN input_return_date DATE, OUT result VARCHAR(100))
label:BEGIN
    DECLARE book_count INT;
    DECLARE reservation_count INT;
    
    -- 是否借阅了该图书
    SELECT COUNT(*) INTO book_count FROM borrow WHERE reader_id = input_reader_id AND book_id = input_book_id AND return_date IS NULL;
    IF book_count = 0 THEN
        SET result = '还书失败：该读者没有借阅该图书。';
        leave label;
    END IF;
    
    -- 更新return_date
    UPDATE borrow SET return_date = input_return_date WHERE reader_id = input_reader_id AND book_id = input_book_id;
    
    -- 判断是否有其他预约
    SELECT COUNT(*) INTO reservation_count FROM reserve WHERE book_id = input_book_id AND take_Date IS NULL;
    
    -- 更新bstatus
    IF reservation_count = 0 THEN
        UPDATE book SET bstatus = 0 WHERE bid = input_book_id; -- 无
    ELSE
        UPDATE book SET bstatus = 2 WHERE bid = input_book_id; 
    END IF;
    
    SET result = '还书成功。';
END //
DELIMITER ;

CALL returnBook('R001', 'B008', '2024-05-10', @result);
SELECT @result;

CALL returnBook('R001', 'B001', '2024-05-10', @result);
SELECT @result;

SELECT bstatus FROM book WHERE bid = 'B001';
SELECT return_date FROM borrow WHERE reader_id = 'R001' AND book_id = 'B001';

-- 6
-- A
delimiter //
drop trigger if exists trigger_reserve_book;
CREATE TRIGGER trigger_reserve_book
AFTER INSERT ON Reserve
FOR EACH ROW
BEGIN
    UPDATE Book
    SET bstatus = 2, reserve_Times = reserve_Times + 1
    WHERE bid = NEW.book_ID;
END//
delimiter ;

drop trigger if exists trigger_borrow_book;
delimiter //
-- B
CREATE TRIGGER trigger_borrow_book
AFTER INSERT ON Borrow
FOR EACH ROW
BEGIN
    UPDATE Book
    SET reserve_Times = reserve_Times - 1
    WHERE bid = NEW.book_ID;
END//
delimiter ;

drop trigger if exists trigger_cancel_reserve;
delimiter //
-- C
CREATE TRIGGER trigger_cancel_reserve
AFTER DELETE ON Reserve
FOR EACH ROW
BEGIN
    DECLARE reserve_count INT;
    DECLARE bookstatus INT default 0;
    SET reserve_count = (SELECT COUNT(*) FROM Reserve WHERE book_ID = OLD.book_ID);
    SELECT bstatus FROM book WHERE bid = OLD.book_ID  INTO bookstatus;
    IF reserve_count = 0 AND bookstatus = 2 THEN
        UPDATE Book
        SET bstatus = 0, reserve_Times = 0
        WHERE bid = OLD.book_ID;
    ELSE
        UPDATE Book
        SET reserve_Times = reserve_Times - 1
        WHERE bid = OLD.book_ID;
    END IF;
END//

delimiter ;

INSERT INTO Reserve (book_ID, reader_ID) VALUES ('B012', 'R001');
SELECT bid,bstatus,reserve_Times FROM book where bid = 'B012';

DELETE FROM Reserve WHERE book_ID = 'B012' AND reader_ID = 'R001';
SELECT bid,bstatus,reserve_Times FROM book where bid = 'B012';