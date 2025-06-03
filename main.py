import math  # стандартная библиотека для математических операций
import itertools  # помогает работать с итерациями (перебор пар вершин)
from collections import deque  # двусторонняя очередь — удобна для BFS обхода графа (граней модели)
from tqdm import tqdm  # отображает прогресс-бар при генерации UV
import matplotlib.pyplot as plt  # для отрисовки развёртки
from matplotlib.patches import Polygon  # рисование 2D-полигонов на графике
from matplotlib.backends.backend_pdf import PdfPages  # сохранение несколько страниц в один файл
from shapely.geometry import Polygon as ShapelyPolygon, Point  # проверка на пересечения, попадания в контур
import threading  # позволяет выполнять отдельные задачи параллельно
import time  # измеряем время выполнения программы
import os  # модуль для работы с путями, файлами и директориями
from PIL import Image  # используется для загрузки и отображения текстур
# Глобальный флаг отладки — если True, включает дополнительный вывод для анализа хода работы
DEBUG = False
def vec_sub(a, b):
    # вычитание векторов: используется для разностей координат двух точек
    # возвращает вектор направления от b к a, необходимый для получения ребра модели
    return [a[i] - b[i] for i in range(len(a))]

def vec_dot(a, b):
    # скалярное произведение: используется для проекций и вычисления углов
    # даёт число, равное сумме попарных произведений координат, применяется:
    # для вычисления косинуса угла между векторами
    # для проекции одной точки на оси другой (compute_face_basis)
    # для проверки направления нормали относительно центра модели (fix_face_orientation)
    return sum(a[i] * b[i] for i in range(len(a)))

def vec_length(a):
    # длина вектора: корень квадратный из скалярного произведения вектора самого на себя
    # нужна для оценки длины ребра, нормализации и статистики скорости обработки
    return math.sqrt(vec_dot(a, a))

def normalize(v):
    # нормализация вектора: получение единичного вектора для удобства вычислений
    # все базисные оси (u, v) и нормали (n) должны быть единичными, иначе искажаются проекции и углы
    l = vec_length(v)
    # при нулевой длине возвращаем исходный вектор, чтобы избежать деления на ноль
    return [vi / l for vi in v] if l != 0 else v

def vec_cross(a, b):
    # векторное произведение: используется для вычисления нормали к плоскости грани
    # результат — вектор, перпендикулярный обоим исходным, длина пропорциональна площади параллелограмма
    # именно через cross(u, temp) в compute_face_basis получаем нормаль n
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    ]

def compute_face_basis(face, vertices):
    # на вход грань (список индексов вершин) и массив вершин модели
    # возвращает систему координат (origin, u, v, n) для развёртки этой грани
    if len(face) < 3:
        # если грань не определена (меньше трёх точек), пропускаем
        return None, None, None, None
    # переводим индексы вершин в реальные 3D-координаты
    pts = [vertices[i] for i in face]
    origin = pts[0]              # выбираем первую точку как начало системы координат грани
    u_raw = vec_sub(pts[1], pts[0])
    u = normalize(u_raw)
    temp = vec_sub(pts[2], pts[0])
    n_raw = vec_cross(u, temp)   # нормаль плоскости грани: u × (вектор к третьей вершине)
    n = normalize(n_raw)
    # вычисляем ортонормальную ось v как n × u
    v_raw = vec_cross(n, u)
    v = normalize(v_raw)
    return origin, u, v, n     # возвращаем начало, две оси плоскости и нормаль

def project_point(P, origin, u, v):
    # проецирует 3D-точку P на плоскость грани и возвращает её 2D-координаты (u, v)
    diff = vec_sub(P, origin)       # перемещаем точку в систему координат грани
    # получаем скалярные координаты вдоль направлений u и v
    return (vec_dot(diff, u), vec_dot(diff, v))

def compute_continuous_unfolding(vertices, raw_faces, max_faces=15):
    # количество граней в исходном списке raw_faces
    num_faces = len(raw_faces)
    # словарь для хранения 2D UV координат каждой грани после развёртки
    face_uv = {}
    # словарь для хранения параметров трансформации каждой грани:
    # угол поворота angle, сдвиг T, базис (origin, u, v, нормаль n)
    face_transform = {}
    # временная структура: для каждого ребра соответствующий список граней, содержащих его
    edge_to_faces = {}

    print("Построение графа смежности...")
    # проходим по всем граням и собираем ребро → грани
    for fi, face in enumerate(raw_faces):
        n = len(face)
        for i in range(n):
            # ребро представлено как упорядоченная кортежем пара индексов вершин
            edge = tuple(sorted((face[i], face[(i + 1) % n])))
            # записываем индекс грани в список для этого ребра
            edge_to_faces.setdefault(edge, []).append(fi)

    # строим список соседей для каждой грани: где сосед определяется общим ребром
    face_neighbors = {i: [] for i in range(num_faces)}
    for edge, fs in edge_to_faces.items():
        # если ребро принадлежит двум граням, они взаимно соседи
        if len(fs) == 2:
            f1, f2 = fs
            face_neighbors[f1].append((f2, edge))
            face_neighbors[f2].append((f1, edge))

    # множество посещённых граней и счётчик компонентов развёртки
    visited = set()
    components = 0

    # используем вывод для отладки и визуализации обработки всех граней
    with tqdm(total=num_faces, desc="Обработка граней") as pbar:
        # перебираем все грани для старта новых компонент развёртки
        for start_face in range(num_faces):
            # если грань уже посещена, пропускаем её
            if start_face in visited:
                continue

            # новая компонента (остров)
            components += 1
            queue = deque([start_face])
            visited.add(start_face)
            comp_count = 1  # сколько граней в текущей компоненте

            # пытаемся построить базис -
            # набор векторов, с помощью которых можно однозначно описать любые точки в некотором векторном пространстве
            # (origin, u, v, n) для стартовой грани
            try:
                origin, u, v, n = compute_face_basis(raw_faces[start_face], vertices)
                # если базис не может быть построен (мало точек) — пропускаем
                if origin is None:
                    continue
            except Exception as e:
                print(f"Ошибка в корневой грани {start_face}: {str(e)}")
                continue

            # сохраняем начальную трансформацию
            face_transform[start_face] = (0.0, (0.0, 0.0), origin, u, v, n)
            # сразу проецируем вершины стартовой грани в 2D UV
            face_uv[start_face] = [
                project_point(vertices[v_idx], origin, u, v)
                for v_idx in raw_faces[start_face]
            ]
            pbar.update(1)  # обновляем прогресс-бар

            # обходим все грани текущей компоненты в ширину
            while queue:
                current = queue.popleft()
                # проверяем, не превысили ли мы максимально разрешённое количество граней
                if max_faces is not None and comp_count >= max_faces:
                    # если да — не добавляем соседей
                    continue

                # извлекаем параметры трансформации текущей грани
                angle_cur, T_cur, cur_origin, cur_u, cur_v, cur_n = face_transform[current]

                # пробегаем по всем соседям через общее ребро
                for neighbor, edge in face_neighbors.get(current, []):
                    # если сосед уже посещён — пропускаем
                    if neighbor in visited:
                        continue

                    # пытаемся построить базис для соседней грани
                    try:
                        nb_origin, nb_u, nb_v, nb_n = compute_face_basis(
                            raw_faces[neighbor], vertices
                        )
                        if nb_origin is None:
                            continue
                    except Exception as e:
                        print(f"Ошибка в грани {neighbor}: {str(e)}")
                        continue

                    # получаем 3D координаты концов ребра
                    e0 = vertices[edge[0]]
                    e1 = vertices[edge[1]]
                    # проецируем эти точки в локальную систему текущей грани
                    cur_e0 = project_point(e0, cur_origin, cur_u, cur_v)
                    cur_e1 = project_point(e1, cur_origin, cur_u, cur_v)

                    # глобальное преобразование: поворот на angle_cur и сдвиг T_cur -
                    # T_cur хранит смещение, которое мы уже применили к грани,
                    # чтобы её локальные (u,v) координаты оказались в той позиции развёртки, где мы их рисуем.
                    cos_a = math.cos(angle_cur)
                    sin_a = math.sin(angle_cur)
                    global_e0 = (
                        cos_a * cur_e0[0] - sin_a * cur_e0[1] + T_cur[0],
                        sin_a * cur_e0[0] + cos_a * cur_e0[1] + T_cur[1]
                    )
                    global_e1 = (
                        cos_a * cur_e1[0] - sin_a * cur_e1[1] + T_cur[0],
                        sin_a * cur_e1[0] + cos_a * cur_e1[1] + T_cur[1]
                    )

                    # проецируем те же точки в локальную систему новой грани
                    nb_e0 = project_point(e0, nb_origin, nb_u, nb_v)
                    nb_e1 = project_point(e1, nb_origin, nb_u, nb_v)

                    # вектор ребра в глобальных координатах текущей развёртки
                    vec_current = (
                        global_e1[0] - global_e0[0],
                        global_e1[1] - global_e0[1]
                    )
                    # вектор ребра в локальных координатах соседней грани
                    vec_neighbor = (
                        nb_e1[0] - nb_e0[0],
                        nb_e1[1] - nb_e0[1]
                    )

                    # angle_diff показывает, на сколько нужно повернуть соседнюю грань,
                    # чтобы её ребро совместилось с текущим ребром
                    angle_diff = (
                        math.atan2(vec_current[1], vec_current[0])
                        - math.atan2(vec_neighbor[1], vec_neighbor[0])
                    )

                    # вычисляем угол между плоскостями граней:
                    phi = abs(angle_diff)
                    if phi > math.pi:
                        # если угол >180°, корректируем до меньшего
                        phi = 2 * math.pi - phi
                    join_angle = math.pi - phi
                    # отбрасываем рычажные углы <45°, чтобы избежать перекрытий
                    if join_angle < math.radians(45):
                        continue

                    # если нормали граней направлены навстречу (dot<0),
                    # добавляем 180° к angle_diff, чтобы правильно развернуть грань
                    dot_n = vec_dot(cur_n, nb_n)
                    if dot_n < 0:
                        angle_diff += math.pi

                    # нормализуем angle_diff в диапазон [-π, π] для стабильности
                    while angle_diff <= -math.pi:
                        angle_diff += 2 * math.pi
                    while angle_diff > math.pi:
                        angle_diff -= 2 * math.pi

                    # матрица поворота в 2D: значения cos и sin
                    cos_diff = math.cos(angle_diff)
                    sin_diff = math.sin(angle_diff)

                    # вращаем координату nb_e0 из системы соседней грани на вычисленный угол,
                    # чтобы получить её положение после поворота в общей системе развёртки
                    rotated_nb_e0 = (
                        cos_diff * nb_e0[0] - sin_diff * nb_e0[1],
                        sin_diff * nb_e0[0] + cos_diff * nb_e0[1]
                    )
                    # вычисляем сдвиг T_nb, чтобы rotated_nb_e0 (ребро соседней грани) совпала с global_e0 (ребро текущей грани)
                    # T_nb — вектор для соседней грани: он говорит,
                    # на сколько и в каком направлении надо «перенести» (трансляцией) всю соседнюю грань,
                    # чтобы она встала правильно рядом с уже развернутой текущей гранью
                    T_nb = (
                        global_e0[0] - rotated_nb_e0[0],
                        global_e0[1] - rotated_nb_e0[1]
                    )

                    # сохраняем все параметры трансформации для соседней грани
                    face_transform[neighbor] = (
                        angle_diff, T_nb, nb_origin, nb_u, nb_v, nb_n
                    )

                    # на основе вычисленных angle_diff и T_nb строим UV координаты:
                    uv_list = []
                    for v_idx in raw_faces[neighbor]:
                        pt = vertices[v_idx]
                        # сначала проекция в локальной системе новой грани
                        local = project_point(pt, nb_origin, nb_u, nb_v)
                        # затем поворот и сдвиг в глобальные - в итоговую развертку
                        rotated = (
                            cos_diff * local[0] - sin_diff * local[1] + T_nb[0],
                            sin_diff * local[0] + cos_diff * local[1] + T_nb[1]
                        )
                        uv_list.append(rotated)
                    face_uv[neighbor] = uv_list

                    # отмечаем грань как посещённую и добавляем в очередь
                    visited.add(neighbor)
                    queue.append(neighbor)
                    comp_count += 1
                    pbar.update(1)

    # финальный отчёт о количестве обработанных граней и компонентов
    print(f"Обработано {len(face_uv)}/{num_faces} граней в {components} компонент(ах)")
    # возвращаем UV координаты и граф смежности ребер
    return face_uv, edge_to_faces


def fix_face_orientation(face_indices, vertices, model_center):
    # получаем базис грани: origin – точка начала, u и v – оси плоскости, n – нормаль
    origin, u, v, n = compute_face_basis(face_indices, vertices)
    # если невозможно вычислить базис (менее трёх вершин) — выходим
    if origin is None:
        return None, None, None, None
    # вычисляем центр грани как среднее всех её вершин
    pts = [vertices[i] for i in face_indices]
    face_center = [sum(pt[j] for pt in pts) / len(pts) for j in range(3)]
    # проверка: вектор от центра грани к центру модели
    # если скалярное произведение с нормалью >0, нормаль направлена внутрь модели
    if vec_dot(n, vec_sub(model_center, face_center)) > 0:
        # инвертируем нормаль, чтобы она всегда смотрела наружу
        n = [-x for x in n]
        # пересчитываем ось v как перекрёстное произведение новой нормали и u
        v = normalize(vec_cross(n, u))
    # возвращаем поправленный базис грани
    return origin, u, v, n

def compute_model_center(vertices):
    # если нет вершин, возвращаем нулевой центр
    if not vertices:
        return [0, 0, 0]
    # считаем среднее по каждой координате из всех вершин модели
    num = len(vertices)
    center = [sum(v[i] for v in vertices) / num for i in range(3)]
    return center

def get_face_groups_2d(faces, uv_coords):
    # строим карту ребро_2D → список граней, которым принадлежит это ребро
    uv_edge_info = {}
    for face_idx, face in enumerate(faces):
        # извлекаем индексы UV координат вершин грани
        uv_inds = [item[1] for item in face]
        n = len(uv_inds)
        for i in range(n):
            # ребро на UV: пара индексов UV
            edge = tuple(sorted((uv_inds[i], uv_inds[(i + 1) % n])))
            uv_edge_info.setdefault(edge, []).append(face_idx)
    # строим граф смежности по UV-ребрам
    graph = {i: set() for i in range(len(faces))}
    for edge, flist in uv_edge_info.items():
        # если ребро разделяют ровно две грани, они являются соседями
        if len(flist) == 2:
            f1, f2 = flist
            graph[f1].add(f2)
            graph[f2].add(f1)
    # находим связные компоненты в этом графе методом DFS
    visited = set()
    groups = []
    for i in range(len(faces)):
        if i not in visited:
            stack = [i]  # используем стек для DFS
            comp = set()
            while stack:
                cur = stack.pop()
                if cur not in visited:
                    visited.add(cur)
                    comp.add(cur)
                    # добавляем всех соседей в стек
                    # DFS быстрее погружает в одно связное «отграниченное» пространство
                    stack.extend(graph[cur] - visited)
            groups.append(comp)
    # возвращаем список групп индексов граней, связанных в 2D
    return groups

def faces_intersect(face_idx1, face_idx2, faces, uv_coords, tol=1e-4):
    # создаём 2D-полигоны для двух граней по их UV-координатам
    poly1 = ShapelyPolygon([uv_coords[uvi] for (_, uvi) in faces[face_idx1]])
    poly2 = ShapelyPolygon([uv_coords[uvi] for (_, uvi) in faces[face_idx2]])
    # вычисляем пересечение и проверяем, превышает ли его площадь порог tol
    inter = poly1.intersection(poly2)
    return inter.area > tol  # возвращаем True, если полигоны реально перекрываются

def get_face_groups_3d(faces):
    # строим граф смежности по 3D-ребрам: каждая грань соединена с соседями по общим рёбрам
    graph = {i: set() for i in range(len(faces))}
    edge_map = {}
    for i, face in enumerate(faces):
        n = len(face)
        for j in range(n):
            # каждое ребро задаётся парами вершин (индексы из пары (vert, uv))
            edge = tuple(sorted((face[j][0], face[(j + 1) % n][0])))
            edge_map.setdefault(edge, []).append(i)
    for edge, face_list in edge_map.items():
        # если два индекса вершин образуют общее ребро ровно двух граней
        if len(face_list) == 2:
            i, j = face_list
            graph[i].add(j)
            graph[j].add(i)
    # присваиваем каждой грани идентификатор связной 3D-компоненты через DFS
    # нам важно собрать все грани одной компоненты без учёта порядка обхода, а не то, насколько быстро найдётся кратчайший путь
    visited = set()
    groups = {}
    group_id = 0
    for i in range(len(faces)):
        if i not in visited:
            stack = [i]
            while stack:
                f = stack.pop()
                if f in visited:
                    continue
                visited.add(f)
                groups[f] = group_id
                stack.extend(graph[f] - visited)
            group_id += 1  # после полной обработки компоненты переходим к следующему ID
    # возвращаем словарь грань → идентификатор 3D-компоненты
    return groups


def get_connected_components(subgroup, faces):
    # строим карту ребро_2D → set граней из subgroup, которым принадлежит это ребро
    uv_edge_map = {}
    for f in subgroup:
        # извлекаем индексы UV координат вершин грани f
        face_uv = [uv_idx for (_, uv_idx) in faces[f]]
        n = len(face_uv)
        for i in range(n):
            # формируем ребро из двух соседних UV индексов
            edge = tuple(sorted((face_uv[i], face_uv[(i+1)%n])))
            # добавляем грань f в множество по этому ребру
            uv_edge_map.setdefault(edge, set()).add(f)

    # строим граф смежности: грань - другие грани, с которыми делит хотя бы одно UV ребро
    graph = {f: set() for f in subgroup}
    for edge, fs in uv_edge_map.items():
        # если ребро принадлежит более чем одной грани, соединяем их в графе
        if len(fs) > 1:
            for f1 in fs:
                for f2 in fs:
                    if f1 != f2:
                        graph[f1].add(f2)

    # находим связные компоненты в этом графе через DFS
    visited = set()
    components = []
    for f in subgroup:
        if f not in visited:
            # стек для обхода
            stack = [f]
            component = set()
            while stack:
                current = stack.pop()
                if current not in visited:
                    visited.add(current)
                    component.add(current)
                    # добавляем всех соседей текущей грани, которых ещё не посещали
                    stack.extend(graph[current] - visited)
            # после полного обхода сохраняем одну компоненту
            components.append(component)
    # возвращаем список наборов граней, которые связаны по UV
    return components

def split_group_by_intersection(group, faces, uv_coords):
    # строим граф конфликтов: ребро пересекает полигоны и грани не должны лежать в одном ряду
    conflict_graph = {face: set() for face in group}
    # перебираем все пары граней внутри исходной группы
    for face1, face2 in itertools.combinations(group, 2):
        # если полигоны в UV пересекаются, помечаем конфликт
        if faces_intersect(face1, face2, faces, uv_coords):
            conflict_graph[face1].add(face2)
            conflict_graph[face2].add(face1)

    # раскладываем грани по строкам, чтобы конфликтующие грани не оказались в одной строке
    row_assignment = {}
    # сортируем грани по числу конфликтов (чем больше, тем раньше распределяем)
    for face in sorted(group, key=lambda f: len(conflict_graph[f]), reverse=True):
        # собираем номера строк, занятых конфликтующими гранями
        used_rows = {row_assignment[other] for other in conflict_graph[face] if other in row_assignment}
        row = 0
        # ищем первый свободный номер строки
        while row in used_rows:
            row += 1
        row_assignment[face] = row

    # группируем грани по строкам
    rows = {}
    for face, row in row_assignment.items():
        rows.setdefault(row, set()).add(face)

    result = []
    # для каждой строки дополнительно делим на связные компоненты по UV (чтобы избежать изолированных групп)
    for row in rows.values():
        components = get_connected_components(row, faces)
        # объединяем все найденные компоненты в один список
        result.extend(components)
    return result

def pack_groups(group_info, page_width, page_height, margin):
    # распределяем прямоугольные области с группами UV на страницы формата page_width×page_height
    pages = []
    current_page = []
    cur_x = margin  # начальная позиция по X с учётом поля
    cur_y = margin  # начальная позиция по Y
    row_max = 0     # высота самой высокой группы в текущем ряду

    for i, info in enumerate(group_info):
        # ширина и высота группы с учётом масштаба
        gw, gh = info['width'], info['height']
        # если новая группа не помещается по ширине, переходим на новую строку
        if cur_x + gw > page_width - margin:
            cur_x = margin
            cur_y += row_max + margin
            row_max = 0
        # если не помещается по высоте текущей страницы, сохраняем страницу и начинаем новую
        if cur_y + gh > page_height - margin:
            pages.append(current_page)
            current_page = []
            cur_x = margin
            cur_y = margin
            row_max = 0
        # размещаем группу с индексом i в текущей позиции
        current_page.append((i, cur_x, cur_y))
        # сдвигаем курсор вправо на ширину группы + отступ
        cur_x += gw + margin
        # обновляем максимальную высоту ряда
        row_max = max(row_max, gh)

    # добавляем последнюю страницу, если она непустая
    if current_page:
        pages.append(current_page)
    return pages

def dihedral_type(edge, f1, f2, vertices, faces, model_center):
    # вычисляем точки ребра в 3D
    e0 = vertices[edge[0]]
    e1 = vertices[edge[1]]
    # вектор ребра (единичный) для проекций нормалей
    d = normalize(vec_sub(e1, e0))
    # центр ребра для определения ориентации диэдра
    midpoint = [(e0[i] + e1[i]) / 2.0 for i in range(3)]

    # извлекаем индексы вершин обеих граней
    face1_indices = [v for (v, uv) in faces[f1]]
    face2_indices = [v for (v, uv) in faces[f2]]
    # корректируем ориентацию нормалей, чтобы они смотрели наружу
    origin1, u1, v1, n1 = fix_face_orientation(face1_indices, vertices, model_center)
    origin2, u2, v2, n2 = fix_face_orientation(face2_indices, vertices, model_center)

    # функция для проекции нормали на плоскость, перпендикулярную ребру
    def project_normal(n):
        dot_val = vec_dot(n, d)
        proj = [n[i] - dot_val * d[i] for i in range(3)]
        return normalize(proj)

    # проецируем обе нормали
    n1p = project_normal(n1)
    n2p = project_normal(n2)
    # вычисляем угол между проекциями нормалей через скалярное произведение
    dot_n = max(-1.0, min(1.0, vec_dot(n1p, n2p)))
    angle = math.acos(dot_n)

    # определяем направление кручения нормалей
    cp = vec_cross(n1p, n2p)
    vec_mid_to_center = vec_sub(model_center, midpoint)
    sign_val = vec_dot(cp, vec_mid_to_center)
    # если sign_val <0, выбираем угол angle, иначе 2π-angle
    if sign_val < 0:
        dihedral = angle
    else:
        dihedral = 2 * math.pi - angle

    if DEBUG:
        # вывод статистики при отладке в консоль
        print(f"DEBUG: Edge {edge}, Faces {f1} & {f2}")
        print(f"       Edge midpoint: {midpoint}")
        print(f"       d: {d}")
        print(f"       n1: {n1}, n1p: {n1p}")
        print(f"       n2: {n2}, n2p: {n2p}")
        print(f"       Angle (deg): {math.degrees(angle):.2f}, Dihedral (deg): {math.degrees(dihedral):.2f}")
        print(f"       sign_val: {sign_val}")
    # возвращаем тип: вогнутый если угол >π, иначе выпуклый
    return "concave" if dihedral > math.pi else "convex"


def draw_unfolding(vertices, uv_coords, faces, edge_info, groups_3d,
                   output_pdf, model_center, scale_percent=100,
                   progress_callback=None, stop_event=None, face_textures=None, tex_map=None, use_textures=False):
    start_time = time.time()  # сохраняем время начала, чтобы собрать статистику в конце

    # если 3D группы граней переданы, используем их, иначе строим группы по UV (2D)
    if groups_3d is not None and len(groups_3d) > 0:
        groups = groups_3d
    else:
        # разбивает все грани на связные группы по UV ребрам
        groups = get_face_groups_2d(faces, uv_coords)

    # разделяем каждую группу по возможным пересечениям, чтобы избежать накладок
    new_groups = []
    for group in groups:
        # проверка прерывания пользователем
        if stop_event and stop_event.is_set():
            raise Exception("Прервано пользователем")
        # результат — список подгрупп без внутренних пересечений
        new_groups.extend(split_group_by_intersection(group, faces, uv_coords))
    groups = new_groups

    # собираем информацию о каждом наборе граней: прямоугольник UV координат
    group_info = []
    max_group_width = max_group_height = 0.0
    for group in groups:
        if stop_event and stop_event.is_set():
            raise Exception("Прервано пользователем")
        # собираем все UV точки для граней группы
        pts = [uv_coords[uv_idx] for f in group for (_, uv_idx) in faces[f]]
        if pts:
            # находим границы по X и Y
            gmin_x = min(p[0] for p in pts); gmax_x = max(p[0] for p in pts)
            gmin_y = min(p[1] for p in pts); gmax_y = max(p[1] for p in pts)
        else:
            # если точек нет — задаём нулевой прямоугольник
            gmin_x = gmax_x = gmin_y = gmax_y = 0.0
        # размеры в UV пространстве
        gw_uv = gmax_x - gmin_x; gh_uv = gmax_y - gmin_y
        # обновляем глобальный максимум, чтобы вычислить масштаб
        max_group_width = max(max_group_width, gw_uv)
        max_group_height = max(max_group_height, gh_uv)
        # сохраняем параметры группы
        group_info.append({
            'group': group,
            'min_x': gmin_x, 'min_y': gmin_y,
            'max_x': gmax_x, 'max_y': gmax_y,
            'width_uv': gw_uv, 'height_uv': gh_uv
        })

    # параметры страницы A4 в мм
    page_width = 210.0; page_height = 297.0; margin = 10.0
    # доступная область внутри полей
    avail_width = page_width - 2 * margin; avail_height = page_height - 2 * margin
    # базовый масштаб, чтобы самая большая группа поместилась на страницу
    if max_group_width < 1e-9 or max_group_height < 1e-9:
        s_base = 1.0  # если группы слишком малы или отсутствуют
    else:
        s_base = min(avail_width / max_group_width, avail_height / max_group_height)
    # окончательный масштаб с учётом процента от пользователя
    s = s_base * (scale_percent / 100.0)

    # фильтруем группы, которые после масштабирования будут слишком малы (<1 мм)
    valid_group_info = []
    for info in group_info:
        if stop_event and stop_event.is_set():
            raise Exception("Прервано пользователем")
        if info['width_uv'] * s < 1.0 or info['height_uv'] * s < 1.0:
            continue  # пропускаем крошечные группы
        # рассчитываем реальные размеры на странице
        info['width'] = info['width_uv'] * s
        info['height'] = info['height_uv'] * s
        valid_group_info.append(info)
    group_info = valid_group_info

    # разбиваем группы по страницам с учётом полей
    pages = pack_groups(group_info, page_width, page_height, margin)
    # сохраняем для каждой группы её страницу и смещение на ней
    group_placement = {}
    for page_num, page in enumerate(pages):
        for (g_idx, ox, oy) in page:
            group_placement[g_idx] = (page_num, ox, oy)

    # для маркировки клапанов: уникальные номера пар рёбер
    global_edge_pair_numbers = {}
    global_pair_number = 1
    tol = 1e-4  # допуск для сравнения UV точек - чтобы не появлялся разрез при раздельных, но очень близких полигонов

    # вспомогательная функция для преобразования одной UV точки
    def transform_uv(uv, info, offset_x, offset_y):
        return (
            (uv[0] - info['min_x']) * s + offset_x,
            (uv[1] - info['min_y']) * s + offset_y
        )

    # преобразует одну грань (список (вершина, uv)) в список точек на странице
    def transform_face(face, info, offset_x, offset_y):
        pts = [transform_uv(uv_coords[uvi], info, offset_x, offset_y)
               for (_, uvi) in face]
        # используем Shapely для очистки возможных самопересечений
        poly = ShapelyPolygon(pts).buffer(0)
        # защитная правка для очень острых треугольников
        if len(pts) == 3:
            area = poly.area; perimeter = poly.length
            if perimeter > 0 and 4 * math.pi * area / (perimeter * perimeter) < 0.1:
                poly = poly.buffer(0.1).buffer(-0.1)
        # возвращаем внешнюю оболочку полигона
        return list(poly.exterior.coords)[:-1] if not poly.is_empty else pts

    # функция для размещения подписи внутри грани ближе к её центру
    def place_label_inside_face(face_points, point, offset_factor=0.3):
        cx = sum(pt[0] for pt in face_points) / len(face_points)
        cy = sum(pt[1] for pt in face_points) / len(face_points)
        return (
            point[0] + offset_factor * (cx - point[0]),
            point[1] + offset_factor * (cy - point[1])
        )

    # для клапанов нужно знать, какие грани находятся в одной 3D компоненте
    face_3d_groups = get_face_groups_3d(faces)
    # размер фигуры в дюймах (1 дюйм = 25.4 мм)
    figsize = (page_width / 25.4, page_height / 25.4)

    # открываем PDF-файл для записи страниц развёртки
    with PdfPages(output_pdf) as pdf:
        total_pages = len(pages)  # общее число страниц, рассчитанное ранее
        for page_num in range(total_pages):  # проходим по каждой странице
            # усли пользователь нажал "стоп", прерываем выполнение
            if stop_event and stop_event.is_set():
                raise Exception("Прервано пользователем")

            # создаём новую страницу с заданными размерами
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_aspect('equal')  # сохраняем одинаковый масштаб по осям
            ax.axis('off')  # отключаем оси (чтобы печатать только картинку)
            ax.set_xlim(0, page_width)  # устанавливаем границы X от 0 до ширины страницы
            ax.set_ylim(0, page_height)  # устанавливаем границы Y от 0 до высоты страницы

            # определяем, какие группы граней попадают на текущую страницу
            groups_on_page = [g for g, (p, _, _) in group_placement.items() if p == page_num]

            # заливка текстурами
            for g in groups_on_page:
                info = group_info[g]  # информация о группе: размеры, границы UV
                ox, oy = group_placement[g][1], group_placement[g][2]  # смещение группы на странице
                for f_idx in info['group']:  # проходим по индексам граней в этой группе
                    # проверяем, нужно ли использовать текстуры и есть ли текстура для этой грани
                    if use_textures and face_textures and face_textures[f_idx] in tex_map:
                        tex_path = tex_map[face_textures[f_idx]]  # путь к файлу текстуры
                        if os.path.exists(tex_path):  # если файл существует
                            # получаем координаты вершин полигона на странице
                            pts_page = transform_face(faces[f_idx], info, ox, oy)
                            xs, ys = zip(*pts_page)  # разделение на X и Y списки

                            # загружаем изображение текстуры и конвертируем в RGBA для прозрачности
                            img = Image.open(tex_path).convert('RGBA')
                            w, h = img.size  # Ширина и высота текстуры в пикселях

                            # собираем UV-координаты для этой грани
                            uv_pts = [uv_coords[uv_idx] for (_, uv_idx) in faces[f_idx]]
                            # преобразуем UV в пиксельные координаты изображения
                            px = [u * w for (u, v) in uv_pts]
                            py = [(1 - v) * h for (u, v) in uv_pts]  # 0 внизу, 1 вверху => переворачиваем

                            # определяем прямоугольник, в который укладываются точки, по минимальным и максимальным пикселям
                            min_px, max_px = int(min(px)), int(max(px))
                            min_py, max_py = int(min(py)), int(max(py))
                            crop = img.crop((min_px, min_py, max_px, max_py))  # Обрезаем текстуру по UV

                            # создаём маску формы полигона внутри обрезанного окна
                            from PIL import ImageDraw
                            mask = Image.new('L', (max_px - min_px, max_py - min_py), 0)
                            draw = ImageDraw.Draw(mask)
                            # точки полигона в координатах обрезанного изображения
                            poly = [((x - min_px), (y - min_py)) for x, y in zip(px, py)]
                            draw.polygon(poly, fill=255)  # Рисуем белый многоугольник на чёрном фоне

                            # применяем маску к обрезанному изображению (отсекаем лишнее)
                            masked = Image.new('RGBA', crop.size)
                            masked.paste(crop, (0, 0), mask)

                            # отображаем полученный кусок текстуры на странице в координатах полигона
                            img_extent = (min(xs), max(xs), min(ys), max(ys))
                            im = ax.imshow(masked, extent=img_extent, zorder=0)

                            # создаём полигон для обрезки рисунка прямо на графике
                            clip_patch = Polygon(pts_page, closed=True)
                            clip_patch.set_transform(ax.transData)
                            im.set_clip_path(clip_patch)

            # отрисовка контуров всех полигонов
            groups_on_page = [g for g, (p, _, _) in group_placement.items() if p == page_num]
            for g_idx in tqdm(groups_on_page, desc="Рисовка полигонов", leave=False):
                if stop_event and stop_event.is_set():
                    raise Exception("Прервано пользователем")
                info = group_info[g_idx]  # параметры группы (границы, UV и т.д.)
                offset_x, offset_y = group_placement[g_idx][1], group_placement[g_idx][2]
                for f in info['group']:
                    pts = transform_face(faces[f], info, offset_x, offset_y)  # 2D-координаты
                    # рисуем полигон без заливки (facecolor 'none') для чётких контуров
                    ax.add_patch(Polygon(pts, closed=True, edgecolor='none', facecolor='none'))

            # рисуем внешние рёбра (те, где только одна грань)
            for edge, entries in edge_info.items():
                if len(entries) == 1:
                    (f1, uv1, uv2) = entries[0]
                    g = None
                    for idx, info in enumerate(group_info):
                        if f1 in info['group']:
                            g = idx; break
                    # если грань находится на этой странице
                    if g is not None and group_placement.get(g, (None,))[0] == page_num:
                        info_face = group_info[g]
                        ox, oy = group_placement[g][1], group_placement[g][2]
                        p_a = transform_uv(uv1, info_face, ox, oy)
                        p_b = transform_uv(uv2, info_face, ox, oy)
                        # рисуем сплошную линию по внешнему ребру
                        ax.plot([p_a[0], p_b[0]], [p_a[1], p_b[1]],
                                color='black', linewidth=0.5, linestyle='-')

            # обработка клапанов и внутренних ребер
            for edge, entries in edge_info.items():
                if len(entries) != 2: continue  # пропускаем не внутренние ребра
                (f1, uv1, uv2), (f2, uv3, uv4) = entries
                g1 = g2 = None
                # на каких группах находятся две грани
                for idx, info in enumerate(group_info):
                    if f1 in info['group']: g1 = idx
                    if f2 in info['group']: g2 = idx

                # проверяем, совпадают ли UV отрезки обеих граней
                uv_coincide = False
                if g1 is not None and g2 is not None and g1 == g2:
                    if (abs(uv1[0] - uv4[0]) < tol and abs(uv1[1] - uv4[1]) < tol and
                        abs(uv2[0] - uv3[0]) < tol and abs(uv2[1] - uv3[1]) < tol):
                        uv_coincide = True

                # выбираем кандидата на клапан: в одной 3D группе и не совпадающие UV
                flapCandidate = (not uv_coincide and f1 in face_3d_groups and
                                 face_3d_groups[f1] == face_3d_groups[f2])
                in_page_f1 = (g1 is not None and group_placement[g1][0] == page_num)
                in_page_f2 = (g2 is not None and group_placement[g2][0] == page_num)

                if flapCandidate:
                    # отрисовка клапана для f1 красным
                    if in_page_f1:
                        info_red = group_info[g1]; ox, oy = group_placement[g1][1], group_placement[g1][2]
                        t1 = transform_uv(uv1, info_red, ox, oy); t2 = transform_uv(uv2, info_red, ox, oy)
                        mid = ((t1[0] + t2[0]) / 2, (t1[1] + t2[1]) / 2)
                        lab = place_label_inside_face(transform_face(faces[f1], info_red, ox, oy), mid)
                        if edge not in global_edge_pair_numbers:
                            global_edge_pair_numbers[edge] = global_pair_number; global_pair_number += 1
                        num = global_edge_pair_numbers[edge]
                        ax.text(lab[0], lab[1], str(num), color='red', fontsize=3,
                                ha='center', va='center', weight='bold')
                        ax.plot([t1[0], t2[0]], [t1[1], t2[1]],
                                color='black', linewidth=0.5, linestyle='-')
                    # отрисовка клапана для f2 синий
                    if in_page_f2:
                        info_blue = group_info[g2]; ox, oy = group_placement[g2][1], group_placement[g2][2]
                        tb1 = transform_uv(uv3, info_blue, ox, oy); tb2 = transform_uv(uv4, info_blue, ox, oy)
                        mid = ((tb1[0] + tb2[0]) / 2, (tb1[1] + tb2[1]) / 2)
                        # вычисляем направление и нормаль для клапана
                        dx = tb2[0] - tb1[0]; dy = tb2[1] - tb1[1]
                        length = math.hypot(dx, dy)
                        if length >= 1e-9:
                            v_dir = (dx / length, dy / length)
                            n_candidate = (-dy / length, dx / length)
                            mid_edge = ((tb1[0] + tb2[0]) / 2, (tb1[1] + tb2[1]) / 2)
                            face_blue_pts = transform_face(faces[f2], info_blue, ox, oy)
                            poly_blue = ShapelyPolygon(face_blue_pts)
                            epsilon = 0.1
                            test_pt = (mid_edge[0] + epsilon * n_candidate[0],
                                       mid_edge[1] + epsilon * n_candidate[1])
                            # инвертируем направление клапана, если оно внутрь полигона
                            if poly_blue.contains(Point(test_pt)):
                                n_candidate = (-n_candidate[0], -n_candidate[1])
                            edge_type = dihedral_type(edge, f1, f2, vertices, faces, model_center)
                            flap_linestyle = '--' if edge_type == "concave" else '-.'
                            # строим многоугольник клапана по трём точкам A, B, C, D
                            perp = (n_candidate[0] * 4.0, n_candidate[1] * 4.0)
                            adjust = min(4.0, length / 2)
                            A = tb1; B = tb2
                            C = (B[0] + perp[0] - v_dir[0] * adjust, B[1] + perp[1] - v_dir[1] * adjust)
                            D = (A[0] + perp[0] + v_dir[0] * adjust, A[1] + perp[1] + v_dir[1] * adjust)
                            ax.plot([B[0], C[0]], [B[1], C[1]], color='black', linewidth=0.5, linestyle='-')
                            ax.plot([C[0], D[0]], [C[1], D[1]], color='black', linewidth=0.5, linestyle='-')
                            ax.plot([D[0], A[0]], [D[1], A[1]], color='black', linewidth=0.5, linestyle='-')
                            ax.plot([A[0], B[0]], [A[1], B[1]], color='black', linewidth=0.5, linestyle=flap_linestyle)
                        lab_blue = place_label_inside_face(transform_face(faces[f2], info_blue, ox, oy), mid, 0.3)
                        if edge not in global_edge_pair_numbers:
                            global_edge_pair_numbers[edge] = global_pair_number; global_pair_number += 1
                        pair_number = global_edge_pair_numbers[edge]
                        ax.text(lab_blue[0], lab_blue[1], str(pair_number),
                                color='blue', fontsize=3, ha='center', va='center', weight='bold')

                # рисуем внутренние рёбра с учётом выпуклости/вогнутости
                if in_page_f1 or in_page_f2:
                    if in_page_f1:
                        g = g1; uv_a, uv_b = uv1, uv2
                    else:
                        g = g2; uv_a, uv_b = uv3, uv4
                    info_face = group_info[g]
                    offset_face = (group_placement[g][1], group_placement[g][2])
                    p_a = transform_uv(uv_a, info_face, offset_face[0], offset_face[1])
                    p_b = transform_uv(uv_b, info_face, offset_face[0], offset_face[1])
                    edge_type = dihedral_type(edge, f1, f2, vertices, faces, model_center)
                    linestyle = '--' if edge_type == "concave" else '-.'
                    ax.plot([p_a[0], p_b[0]], [p_a[1], p_b[1]],
                            color='black', linewidth=0.5, linestyle=linestyle)

            # сохраняем отрисованную страницу в PDF
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            # обновляем прогресс-бар через callback
            if progress_callback:
                if stop_event and stop_event.is_set():
                    raise Exception("Прервано пользователем")
                percent = int((page_num + 1) / total_pages * 100)
                progress_callback(percent)

            # выводим информацию о сохранении страницы
            print(f"Развёртка сохранена: {output_pdf}, страниц: {total_pages}")

            # после последней страницы собираем и выводим статистику в консоль
            total_time = time.time() - start_time
            total_polygons = len(faces)
            num_groups = len(groups)
            avg_speed = total_polygons / total_time if total_time > 0 else 0

            print("\n📊 Статистика выполнения:")
            print(f"|{'Параметр':<25}|{'Значение':>15}|")
            print(f"|{'-' * 25}|{'-' * 15}|")
            print(f"|Общее время обработки: |{total_time:>15.2f} сек|")
            print(f"|Полигонов обработано:  |{total_polygons:>15}|")
            print(f"|Комплектов развёрток:  |{num_groups:>15}|")
            print(f"|Скорость обработки:    |{avg_speed:>15.2f} полиг/сек|")
            print(f"|Страниц сгенерировано: |{total_pages:>15}|")


def parse_obj(file_path, force_uv=False, max_faces=15):
    # приводим путь к стандартному формату и определяем папку, где лежит модель
    file_path = os.path.normpath(file_path)
    base_dir = os.path.dirname(file_path)

    # подготовка структур для данных из OBJ
    vertices = []       # список 3D-вершин
    uv_coords = []      # список UV-координат
    faces = []          # список граней с UV-индексами
    raw_faces = []      # грани без UV (для генерации)
    edge_info = {}      # информация об UV-ребрах
    edge_to_faces_3d = {}  # информация о 3D-ребрах

    # храним, какой материал (и текстура) используется у каждой грани
    tex_map = {}          # материал -> путь к файлу текстуры
    face_textures = []    # список материалов для каждой грани

    # сначала ищем в OBJ ссылку на MTL
    mtl_lib = None
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.lower().startswith('mtllib '):
                mtl_lib = line.split(maxsplit=1)[1].strip()
                break

    # если MTL найден, парсим его, чтобы узнать пути к текстурам
    if mtl_lib:
        mtl_path = os.path.join(base_dir, os.path.normpath(mtl_lib))
        if os.path.exists(mtl_path):
            curr_mat = None
            with open(mtl_path, 'r', encoding='utf-8') as mf:
                for ml in mf:
                    parts = ml.strip().split()
                    if not parts:
                        continue
                    key = parts[0].lower()
                    if key == 'newmtl':
                        curr_mat = parts[1]  # меняем текущий материал
                    elif key == 'map_kd' and curr_mat:
                        tex_file = ' '.join(parts[1:]).strip('"')  # относительный путь
                        # ищем файл: сначала по тому, как написано, затем только имя
                        tex_rel = os.path.normpath(tex_file)
                        tex_path = os.path.join(base_dir, tex_rel)
                        if not os.path.exists(tex_path):
                            # если не нашли напрямую, ищем по имени файла на всём диске
                            basename = os.path.basename(tex_rel)
                            for root, dirs, files in os.walk(base_dir):
                                if basename in files:
                                    tex_path = os.path.join(root, basename)
                                    break
                        # сохраняем, только если существует файл
                        if os.path.exists(tex_path):
                            tex_map[curr_mat] = tex_path

    # теперь считываем сам OBJ
    curr_mat = None
    face_idx = 0
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line.startswith('v '):
                # добавляем 3D-вершину в список
                vertices.append(list(map(float, line.split()[1:])))
            elif line.startswith('vt '):
                # добавляем UV-координату
                uv_coords.append(list(map(float, line.split()[1:])))
            elif line.startswith('usemtl '):
                # сохраняем имя материала для следующих граней
                curr_mat = line.split(maxsplit=1)[1].strip()
            elif line.startswith('f '):
                parts = line.split()[1:]
                # если есть UV и не форсируем генерацию, парсим с UV
                if '/' in parts[0] and uv_coords and not force_uv:
                    face = []
                    verts = []
                    for part in parts:
                        idx = part.split('/')
                        v_idx = int(idx[0]) - 1
                        uv_idx = int(idx[1]) - 1 if len(idx)>1 and idx[1] else 0
                        face.append((v_idx, uv_idx))
                        verts.append(v_idx)
                    faces.append(face)
                    face_textures.append(curr_mat)  # материал для этой грани
                    # сохраняем связку ребро-грань для отрисовки ребра
                    for i in range(len(verts)):
                        e = tuple(sorted((verts[i], verts[(i+1)%len(verts)])))
                        uv1 = uv_coords[face[i][1]]
                        uv2 = uv_coords[face[(i+1)%len(verts)][1]]
                        edge_info.setdefault(e, []).append((face_idx, uv1, uv2))
                    face_idx += 1
                else:
                    # без UV: собираем raw_faces
                    raw = [int(p.split('/')[0]) - 1 for p in parts]
                    raw_faces.append(raw)
                    for i in range(len(raw)):
                        e = tuple(sorted((raw[i], raw[(i+1)%len(raw)])))
                        edge_to_faces_3d.setdefault(e, []).append(face_idx)
                    face_idx += 1

    # если не нашли ни одной грани ни с UV, ни без — файл не содержит валидной модели
    if not raw_faces and not faces:
        raise ValueError("Некорректный файл, объект не найден")

    # если нужно форсировать генерацию UV или они отсутствуют в файле
    if force_uv or (not uv_coords and raw_faces):
        # генерируем UV развёртку для raw_faces
        face_uv, _ = compute_continuous_unfolding(vertices, raw_faces, max_faces=max_faces)

        # сброс старых UV координат
        uv_coords = []
        new_faces = []
        uv_map = {}  # карта уникальных UV→новый индекс

        # проходим по каждой raw_face и соответствующим сгенерированным UV
        for f_idx, face in enumerate(raw_faces):
            uv_face = []
            new_face = []
            n = len(face)
            for i, v_idx in enumerate(face):
                uv_pt = face_uv[f_idx][i]  # сгенерированные UV координаты
                uv_val = (round(uv_pt[0], 6), round(uv_pt[1], 6))  # округляем для стабильности ключей
                current_edge = tuple(sorted((face[i], face[(i + 1) % n])))
                # если ребро внутреннее (два соседних raw_faces), ключ=UV; иначе уникализируем ключ по грани
                if current_edge in edge_to_faces_3d and len(edge_to_faces_3d[current_edge]) == 2:
                    key = uv_val
                else:
                    key = (f_idx, i, uv_val[0], uv_val[1])
                # назначаем или извлекаем индекс UV для этой точки
                if key in uv_map:
                    uv_index = uv_map[key]
                else:
                    uv_index = len(uv_coords)
                    uv_coords.append(list(uv_val))
                    uv_map[key] = uv_index
                # записываем вершину с новым UV индексом
                uv_face.append((v_idx, uv_index))
                new_face.append((v_idx, uv_index))
            new_faces.append(new_face)

        # заменяем faces на новый список со сгенерированными UV
        faces = new_faces

        # восстанавливаем edge_info для новых faces, чтобы правильно связать новые полигоны
        face_idx = 0
        for face in faces:
            face_verts = [v_idx for (v_idx, uv_idx) in face]
            n = len(face_verts)
            for i in range(n):
                v1 = face_verts[i]
                v2 = face_verts[(i + 1) % n]
                e = tuple(sorted((v1, v2)))
                uv1 = uv_coords[face[i][1]]
                uv2 = uv_coords[face[(i + 1) % n][1]]
                edge_info.setdefault(e, []).append((face_idx, uv1, uv2))
            face_idx += 1

    # возвращаем все собранные данные для дальнейшей развёртки
    return vertices, uv_coords, faces, edge_info, face_textures, tex_map

import tkinter as tk                          # стандартный модуль для создания GUI
from tkinter import filedialog, messagebox, ttk  # диалоговые окна и стилизованные виджеты

def launch_gui():
    # вспомогательные функции
    def show_max_faces_help():
        messagebox.showinfo(
            "Подсказка по Max Faces",
            "Это число максимального количества полигонов в одной детали.\n"
            "Чем больше число вы выберете — тем сложнее деталь будет для вырезания и склейки, но тем меньше будет страниц.\n"
            "Оптимальное число — 15."
        )

    def save_file(entry):
        p = filedialog.asksaveasfilename(defaultextension=".pdf",
                                         filetypes=[("PDF files", "*.pdf")])
        if p:
            entry.delete(0, tk.END)
            entry.insert(0, p)

    def select_file(entry):
        path = filedialog.askopenfilename(filetypes=[("OBJ files", "*.obj")])
        if not path:
            return
        entry.delete(0, tk.END)
        entry.insert(0, path)
        # анализ UV и текстур
        has_vt = False; cnt = 0; has_tex = False
        mtl_candidates = []
        try:
            with open(path, 'r') as f:
                for ln in f:
                    if ln.startswith('vt '): has_vt = True
                    if ln.startswith('f '): cnt += 1
                    if ln.lower().startswith('mtllib '):
                        mtl_candidates.append(ln.split(maxsplit=1)[1].strip())
        except:
            lbl_uv.config(text="Ошибка при чтении файла", foreground="red")
            return
        base = os.path.dirname(path)
        for m in mtl_candidates:
            mp = os.path.join(base, os.path.normpath(m))
            if os.path.exists(mp):
                with open(mp, 'r') as mf:
                    for l in mf:
                        if l.lower().startswith('map_kd '):
                            has_tex = True
                            break
        # настраиваем комбобоксы
        max_f = min(cnt, 30)
        combo_max_faces['values'] = [str(i) for i in range(1, max_f+1)] + ["Без ограничений"]
        combo_max_faces.set(str(min(15, max_f)))
        if has_vt:
            lbl_uv.config(text="UV найдены", foreground="green")
            combo_uv['values'] = ("Использовать UV из файла", "Генерировать UV")
            combo_uv.current(0)
            combo_max_faces.config(state='disabled')
        else:
            lbl_uv.config(text="UV не найдены", foreground="red")
            combo_uv['values'] = ("Генерировать UV",)
            combo_uv.current(0)
            combo_max_faces.config(state='readonly')
        # текстуры
        if has_tex:
            chk_tex.config(state='normal')
        else:
            chk_tex.config(state='disabled')
            use_tex_var.set(False)

    # создаём основное окно приложения
    root = tk.Tk()
    root.title("Генератор развёрток 3D-моделей")

    frm = ttk.Frame(root, padding=10)
    frm.grid()

    # OBJ-файл
    ttk.Label(frm, text="OBJ-файл:").grid(column=0, row=0, sticky="w")
    entry_obj = ttk.Entry(frm, width=50)
    entry_obj.grid(column=1, row=0)
    btn_obj = ttk.Button(frm, text="Выбрать...", command=lambda: select_file(entry_obj))
    btn_obj.grid(column=2, row=0)
    lbl_uv = ttk.Label(frm, text="UV не проверены")
    lbl_uv.grid(column=1, row=1, columnspan=2, sticky="w")

    # UV-режим
    ttk.Label(frm, text="Режим UV:").grid(column=0, row=2, sticky="w")
    combo_uv = ttk.Combobox(frm, state="readonly", width=25)
    combo_uv.grid(column=1, row=2, columnspan=2, sticky="we")
    combo_uv.bind('<<ComboboxSelected>>', lambda e: combo_max_faces.config(state='readonly' if 'генерировать' in combo_uv.get().lower() else 'disabled'))

    # Max faces
    ttk.Label(frm, text="Максимально полигонов:").grid(column=0, row=3, sticky="w")
    combo_max_faces = ttk.Combobox(frm, state="disabled", width=5)
    combo_max_faces.grid(column=1, row=3, sticky="w")
    help_btn = ttk.Button(frm, text="?", width=2, command=show_max_faces_help)
    help_btn.grid(column=2, row=3, sticky="w")

    # PDF output
    ttk.Label(frm, text="Сохранить как:").grid(column=0, row=4, sticky="w")
    entry_pdf = ttk.Entry(frm, width=50)
    entry_pdf.grid(column=1, row=4)
    btn_pdf = ttk.Button(frm, text="Обзор...", command=lambda: save_file(entry_pdf))
    btn_pdf.grid(column=2, row=4)

    # масштаб
    ttk.Label(frm, text="Масштаб (%):").grid(column=0, row=5, sticky="w")
    spin_scale = tk.Spinbox(frm, from_=1, to=100, width=5)
    spin_scale.grid(column=1, row=5, sticky="w")
    spin_scale.delete(0, tk.END)
    spin_scale.insert(0, "100")

    # чекбокс текстур
    use_tex_var = tk.BooleanVar()
    chk_tex = ttk.Checkbutton(frm, text="Использовать текстуры", variable=use_tex_var)
    chk_tex.grid(column=0, row=6, columnspan=3, sticky='w')
    chk_tex.config(state='disabled')

    # кнопки и прогресс
    btn_run = ttk.Button(frm, text="Создать развёртку")
    btn_run.grid(column=1, row=7, padx=5)
    btn_stop = ttk.Button(frm, text="Стоп", state="disabled")
    btn_stop.grid(column=2, row=7)
    prog = ttk.Progressbar(frm, orient='horizontal', mode='determinate', length=200)
    prog.grid(column=1, row=8, columnspan=2, pady=5)

    # запуск развёртки
    def run_unfolding():
        obj = entry_obj.get()
        out = entry_pdf.get()
        if not obj or not out:
            messagebox.showerror("Ошибка", "Укажите пути")
            return
        btn_run.config(state='disabled')
        btn_stop.config(state='normal')
        prog['value'] = 0
        prog.config(maximum=100)
        stop_event = threading.Event()
        btn_stop.config(command=stop_event.set)
        def progress_cb(p):
            prog['value'] = p
        def bg():
            try:
                # определяем, генерируем UV или используем существующие
                force_uv = 'генерировать' in combo_uv.get().lower()
                mf = combo_max_faces.get()
                mf = None if mf == "Без ограничений" else (int(mf) if mf.isdigit() else None)
                # парсим OBJ/MTL на основе force_uv и max_faces
                verts, uvcs, faces, edge_info, face_tex, tex_map = parse_obj(
                    obj, force_uv=force_uv, max_faces=mf if mf else 15)
                # центр модели важен для расчёта направления клапанов
                center = compute_model_center(verts)
                # масштабирование развёртк
                scale = float(spin_scale.get())
                # основная отрисовка развёртки в PDF
                draw_unfolding(
                    verts, uvcs, faces, edge_info, None,
                    out, center, scale_percent=scale,
                    progress_callback=progress_cb,
                    stop_event=stop_event,
                    face_textures=face_tex,
                    tex_map=tex_map,
                    use_textures=use_tex_var.get()
                )
                # если пользователь не нажал "Стоп", показываем успех
                if not stop_event.is_set():
                    messagebox.showinfo("Успех", f"Сохранено: {out}")
            except Exception as e:
                # любая ошибка выводится всплывающим окном
                messagebox.showerror("Ошибка", str(e))
            finally:
                # разблокируем кнопки GUI в любом случае
                btn_run.config(state='normal')
                btn_stop.config(state='disabled')
        # запускаем фоновую задачу и возвращаем управление GUI
        threading.Thread(target=bg, daemon=True).start()
    # Привязываем функцию к кнопке запуска и запускаем главный цикл GUI
    btn_run.config(command=run_unfolding)
    root.mainloop()
if __name__ == '__main__':
    launch_gui()  # точка входа при запуске скрипта напрямую
