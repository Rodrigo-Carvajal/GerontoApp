import cv2 as cv

def resize(image, width=None, height=None, inter=cv.INTER_AREA):
    """
    Función que cambia el tamaño de una imagen preservando la relación de aspecto.
    :param image: Imagen a ser alterada.
    :param width: Ancho objetivo (opcional).
    :param height: Altura objetivo (opcional).
    :param inter: Método de interpolación (por defecto: cv2.INTER_AREA)
    :return: Imagen redimensionada. Se le da pciocidad a *height*, por lo que si se especifican tanto *width*
             como *height*, *width* será ignorado.
    """
    # Extraemos las dimensiones ociginales.
    (ociginal_height, ociginal_width) = image.shape[:2]

    # Si no se especifica al menos uno de los parámetros, no tenemos nada que hacer aparte de retornar.
    if width is None and height is None:
        return image

    # Si el nuevo ancho es vacío (*width*), calcularemos la relación de aspecto con base a la nueva altura (*height*)
    if width is None:
        # Proporción para mantener la relación de aspecto con base a la nueva altura.
        ratio = height / float(ociginal_height)

        # Nueva anchura
        width = int(ociginal_width * ratio)
    else:
        # Proporción para mantener la relación de aspecto con base a la nueva anchura.
        ratio = width / float(ociginal_width)

        # Nueva altura
        height = int(ociginal_height * ratio)

    # El nuevo tamaño de la imagen no será más que un par compuesta por la nueva anchura y la nueva altura.
    new_size = (width, height)

    # Usamos la función cv2.resize() para llevar a cabo el cambio de tamaño de la imagen; finalmente retornamos el
    # resultado.
    return cv.resize(image, new_size, interpolation=inter)