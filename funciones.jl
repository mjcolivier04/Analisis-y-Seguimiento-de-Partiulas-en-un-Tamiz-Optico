using VideoIO
using Plots
using Images
using LaTeXStrings
using Measures


#Estructura que usaremos en el análisis
mutable struct Partícula
    ID::Int #Un número que simplemente sirve para diferenciar entre partículas
    pos_inicial::CartesianIndex{2} #La primera posición
    trayectoria::Vector{CartesianIndex{2}} #Puntos que conforman la trayectoria
    trayectoria_frames::Vector{Int64} #Frames en los cuales sucedieron esos puntos
    pos_final::CartesianIndex{2} #La última posición registrada
end

"""
Esta función aplica un filtro gris a la imagen y clasifica cada pixel de acuerdo a su intensidad relativa,
regresando una imagen binaria como resultado (el pixel es parte de una partícula o no). 
"""
function procesar_imagen!(img::PermutedDimsArray{RGB{N0f8}, 2, (2, 1), (2, 1), Matrix{RGB{N0f8}}})
    img = Gray.(img) #Aplica un filtro de escala de grises para hacer un gradiente de color fácilemente
    #Se vuelve blanco o negro dependiendo de la intensidad de grises (la tolerancia se determinó por prueba y error)
    tol = 0.1
    for x in 1:tam_x
        for y in 1:tam_y
            img[x, y] <= tol ? img[x, y] = 1 : img[x, y] = 0 #Si es muy claro, se hace blanco y si no se hace negro
        end
    end
    return img
end

"""
Esta función halla las posiciones de todas las partículas presentes en un frame, regresando un
vector de CartesianIndex{2} como resultado.
"""
function encontrar_partículas_1_frame!(img::PermutedDimsArray{RGB{N0f8}, 2, (2, 1), (2, 1), Matrix{RGB{N0f8}}})
    #Se pre-procesa la imagen
    img = procesar_imagen!(img)
    #Se encuentran blobs o "bolitas" usando Laplacian of Gaussian (es un kérnel estándar pero con σ ajustable)
    blobs = Images.blob_LoG( img , [1.6*√2, 1.60001*√2])
    #locs es un vector de CartesianIndex de las posiciones de los blobs si acaso no están en la frontera
    locs = [b.location for b in blobs if !in(b.location, frontera)]
    return locs #Vector de CartesianIndex que tiene las posiciones de las partículas
end

"""
Esta función saca la distancia entre dos CartesianIndex{2} como una distancia euclideana.
"""
dist_índices(ind1::CartesianIndex{2}, ind2::CartesianIndex{2}) = sqrt((ind1[1] - ind2[1])^2 + (ind1[2] - ind2[2])^2)

"""
Esta función dada una posición CartesianIndex{2}, halla la más cercana posible dado un vector de CartesianIndex{2}
que son los candidatos. Si el posible candidato está en la frontera, regresa 0.
"""
function partícula_más_cercana(ind::CartesianIndex{2}, locs::Vector{CartesianIndex{2}})
    return sort(locs, by = x -> dist_índices(ind, x))[1]
end

"""
Esta función itera sobre cada frame del video y se encarga de identificar quién es la misma partícula en el siguiente
frame. Con eso, va almacenando las trayectorias seguidas por las partículas hasta que se unan entre ellas, en cuyo caso
las 2 o más trayectorias se vuelven sólo una o hasta que alguna de las partículas que sigue pasa por la frontera,
en cuyo caso se corta el seguimiento y se manda la trayectoria que se manda a un array "histórico" de partículas
que alguna vez aparecieron pero se dejaron de seguir.
"""
function seguir_partículas()
    global video = VideoIO.load("videoparticulas.mp4");
    #Tamaño de frames (se asume constante a lo largo del video)
    global tam_x, tam_y = size(video[1])[1], size(video[1])[2]; #renglones y columnas respectivamente
    #Se define la frontera como una región de 10 pixeles de ancho en cada lado de la imagen donde no se detectarán
    #las partículas, esto con la base de que una partícula tiene un diámetro de ∼5 pixeles y nunca viajan más de
    #eso de una frame a otro.
    global frontera = [[CartesianIndex(x, y) for x in [1:10; tam_x-9:tam_x] for y in 1:tam_y];
            [CartesianIndex(x, y) for x in 10:tam_x-9 for y in [1:10; tam_y-9:tam_y]]];
    
    partículas = Partícula[] #Vector vacío de partículas que por supuesto tiene ID, pos_inicial, trayectoria y pos_final
    #Lo llenamos con aquellas partículas presentes en el primer frame
    for (ID, pos) in enumerate(encontrar_partículas_1_frame!(video[1]))
        push!(partículas, Partícula(ID, pos, [pos], [1], pos)) #La posición inicial y final coinciden por ahora
    end

    #Ahora, revisamos frame a frame quién es el candidato más cercano a cada partícula y vamos actualizando
    #las trayectorias mientras nos quedemos dentro de la zona segura.
    partículas_que_desaparecieron = Partícula[] #Aquellas partículas que dejaron de aparecer o se fueron a la frontera
    for frame in 2:length(video) #Iteramos sobre cada frame del video
        candidatos_anteriores = encontrar_partículas_1_frame!(video[frame-1]) #Las posiciones de partícula un frame antes
        candidatos = encontrar_partículas_1_frame!(video[frame]) #Las posiciones de ahora
        partículas_por_borrar = Int64[]
        for i in 1:length(partículas) #Revisamos aquellas que existieron desde antes de este frame
            partícula = partículas[i]
            más_cercana = partícula_más_cercana(partícula.trayectoria[end], candidatos) #El mejor candidato
            if in(más_cercana, frontera) #Si la más cercana está en la frontera, la dejamos de seguir
                partícula.pos_final = partícula.trayectoria[end] #Su última posición se actualiza por última vez
                push!(partículas_que_desaparecieron, partícula) #Lo mandamos a las partículas que ya no se analizarán
                push!(partículas_por_borrar, i) #Lo borramos de las partículas por analizar
            elseif dist_índices(partícula.trayectoria[end], más_cercana) > 5 #Muy lejos, no puede ser la misma
                #En este caso no puede ser la misma partícula porque está demasiado lejos, más bien desapareció
                partícula.pos_final = partícula.trayectoria[end] #Su última posición se actualiza por última vez
                push!(partículas_que_desaparecieron, partícula) #Lo mandamos a las partículas que ya no se analizarán
                push!(partículas_por_borrar, i) #Lo borramos de las partículas por analizar
            else #Si no pasa lo anterior, se actualiza la trayectoria como si nada y se sigue
                push!(partícula.trayectoria, más_cercana)
                push!(partícula.trayectoria_frames, frame)
                partícula.pos_final = partícula.trayectoria[end]
            end
        end
        #Una vez revisadas todas las partículas que estaban ahí desde antes, proseguimos a revisar si hay alguna nueva
        #para ello diremos que una partícula es nueva si está en candidatos (de este frame) pero su distancia a todos
        #los candidatos_anteriores es demasiado grande como para que sea una de esas partículas de antes.
        #Siendo así, la agregamos a la lista de partículas por analizar.
        último_ID = max(partículas[end].ID, partículas_que_desaparecieron[end].ID)
        for cand in candidatos
            if !in(cand, frontera)
                if prod([dist_índices(cand, ind) for ind in candidatos_anteriores] .>= 5)
                    push!(partículas, Partícula(último_ID + 1, cand, [cand], [frame], cand))
                    último_ID += 1
                end
            end
        end
        deleteat!(partículas, partículas_por_borrar)
    end
    #Como realmente no importa cuáles se quedaron o cuáles desaparecieron, regresaremos todas las posibles trayectorias
    #que hayan sucedido en algún momento como un vector de objetos Partícula
    return vcat(partículas, partículas_que_desaparecieron) #No descarta nada, ni siquiera trayectorias de un sólo punto
end

"""
Esta función recibe el seguimiento de las partículas (vector de Partículas que ya tiene sus trayectorias guardadas)
y regresa el desplazamiento cuadrático medio como una lista de desplazamientos hasta el enésimo frame.
"""
function MSD(partículas::Vector{Partícula})
    desplazamientos = Float64[]
    for frame in 2:length(video)
        #Ahora filtramos aquellas partículas cuya trayectoria tenga más de un punto, para poder hacer cálculos
        #de distancia recorrida. Y también filtramos por aquellas que hayan aparecido antes del frame a estudiar 
        #en cuestión (si aún no han aparecido o justo acaban de aparecer, no contribuyen a la distancia hasta ahora).
        filtro = findall(x -> (length(x.trayectoria) > 1 && x.trayectoria_frames[1] < frame), partículas)
        partículas_filtradas = partículas[filtro]
        msd = 0
        for p in partículas_filtradas
            if p.trayectoria_frames[end] <= frame
                msd += dist_índices(p.trayectoria[1], p.trayectoria[end])^2
            elseif p.trayectoria_frames[end] > frame
                msd += dist_índices(p.trayectoria[1],
                                    p.trayectoria[length(p.trayectoria_frames) - (p.trayectoria_frames[end] - frame) + 1])^2
            end
        end
        msd /= length(partículas_filtradas)
        push!(desplazamientos, msd)
    end
    
    return desplazamientos
end

"""
Esta función grafica el desplazamiento cuadrático medio como función del tiempo.
"""
function graficar_MSD(Y1)
    X = 2:length(video)
    p1 = plot(X, Y1, title=L"$\langle x^2 \rangle$ vs $t$", xlabel = L"Tiempo $t$ [número de frames]",
              ylabel = L"Desplazamiento cuadrático
    medio $\langle x^2 \rangle(t)$", key = false)
    return p1
end

"""
Esta función grafica el desplazamiento cuadrático medio dividido por el tiempo como función del tiempo. Y con ello
calcula el coeficiente de difusión
"""
function difusión(msd)
    X = 2:length(video)
    Y = msd ./ X
    coeficiente_de_difusión = Y[end]
    p1 = plot(X, Y, title=L"$\frac{\langle x^2 \rangle}{t}$ vs $t$", xlabel = L"Tiempo $t$ [número de frames]",
              ylabel = L"Desplazamiento cuadrático
    medio sobre $t$ $\frac{\langle x^2 \rangle(t)}{t}$", left_margin=2mm, leg=false)
    return p1, coeficiente_de_difusión #difusión[1] es la gráfica y difusión[2] es el valor
end

function graficar_trayectoria(track, m)    
    A = zeros(tam_x, tam_y)
    A[track[m].trayectoria] .= 1
    Imagen = Gray.(A)
    save("trayectoria.png", Imagen)
end