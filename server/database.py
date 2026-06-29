import json
from sqlalchemy import create_engine, Column, Integer, String, Float, Text
from sqlalchemy.orm import declarative_base, sessionmaker


engine = create_engine("sqlite:///stylemate.db", echo=False)

Base= declarative_base()

class Combination(Base):

    __tablename__ = "combinations"

    id = Column(Integer, primary_key= True, autoincrement= True)
    top = Column(String)
    bottom = Column(String)
    liked = Column(Integer)
    combo_vector = Column(Text)

Base.metadata.create_all(engine)
Session = sessionmaker(bind = engine)


def save_combination(top, bottom, combo_vector, liked= 0):

    session= Session()
    existing = session.query(Combination).filter_by(top= top, bottom=bottom).first()

    if existing is None:
        new_combo= Combination(
            top=top,
            bottom=bottom,
            liked=liked,
            combo_vector= json.dumps(combo_vector)

        )
        session.add(new_combo)
        session.commit()
        print(f"DB: Saved new combination → {top} + {bottom}")
    else:
        print(f"DB: Combination already exists → {top} + {bottom}")

    session.close()        


def update_liked(top, bottom):

    session= Session()
    combo = session.query(Combination).filter_by(top= top, bottom=bottom).first()

    if combo is not None:
        combo.liked =1
        session.commit()
        print(f"DB: Updated liked=1 → {top} + {bottom}")
    else:
        # If somehow the combination wasn't saved yet, save it now with liked=1
        save_combination(top, bottom, [], liked=1)

    session.close()


def get_all_combinations():

    session= Session()
    all_rows = session.query(Combination).all()

    result = []
    for row in all_rows:
        result.append({
            "top":          row.top,
            "bottom":       row.bottom,
            "liked":        row.liked,
            "combo_vector": json.loads(row.combo_vector) if row.combo_vector else []
        })

    session.close()
    return result


def count_combinations():

    session= Session()
    total= session.query(Combination).count()
    liked = session.query(Combination).filter_by(liked= 1).count()

    session.close()
    return total, liked
