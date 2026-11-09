import csv
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

SEMENTE = 42
BATCH = 128
TAXA = 1e-3
EPOCAS_BUSCA = 15
EPOCAS_RETREINO = 40
PACIENCIA = 5
N_VAL = 6000
RAIZ = Path(__file__).resolve().parent
DATASET = RAIZ / "Dataset"
RESULTADOS = RAIZ / "resultados"
LARGURAS = ((256, 128), (512, 256))
DIMENSOES = (2, 8)
NOMES = [
    "Camiseta",
    "Calça",
    "Pulôver",
    "Vestido",
    "Casaco",
    "Sandália",
    "Camisa",
    "Tênis",
    "Bolsa",
    "Bota",
]


def definir_semente(semente):
    random.seed(semente)
    np.random.seed(semente)
    torch.manual_seed(semente)


def carregar_csv(caminho):
    """Lê o CSV do Fashion-MNIST. Devolve pixels em [0, 1] e o rótulo."""
    tabela = np.loadtxt(caminho, delimiter=",", skiprows=1, dtype=np.float32)
    rotulos = tabela[:, 0].astype(np.int64)
    pixels = tabela[:, 1:] / 255.0
    return pixels, rotulos


def separar_validacao(pixels, rotulos, n_val, semente):
    """Separa n_val exemplos do treino, com a mesma quantidade de cada classe."""
    gerador = np.random.default_rng(semente)
    por_classe = n_val // 10
    indice_treino = []
    indice_val = []
    for classe in range(10):
        indices = np.where(rotulos == classe)[0]
        escolhidos = gerador.choice(indices, size=por_classe, replace=False)
        indice_val.append(escolhidos)
        indice_treino.append(np.setdiff1d(indices, escolhidos))
    return np.concatenate(indice_treino), np.concatenate(indice_val)


def carregar_dados():
    x_todo, y_todo = carregar_csv(DATASET / "fashion-mnist_train.csv")
    x_teste, y_teste = carregar_csv(DATASET / "fashion-mnist_test.csv")
    i_treino, i_val = separar_validacao(x_todo, y_todo, N_VAL, SEMENTE)
    dados = {
        "x_treino": x_todo[i_treino],
        "y_treino": y_todo[i_treino],
        "x_val": x_todo[i_val],
        "y_val": y_todo[i_val],
        "x_teste": x_teste,
        "y_teste": y_teste,
    }
    return dados


class VAE(nn.Module):
    """MLP simétrica. O encoder devolve μ e log σ²; o decoder termina em sigmoide."""

    def __init__(self, largura, dim_latente):
        super().__init__()
        h1, h2 = largura
        self.largura = largura
        self.dim_latente = dim_latente
        self.fc1 = nn.Linear(784, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc_mu = nn.Linear(h2, dim_latente)
        self.fc_logvar = nn.Linear(h2, dim_latente)
        self.fc3 = nn.Linear(dim_latente, h2)
        self.fc4 = nn.Linear(h2, h1)
        self.fc5 = nn.Linear(h1, 784)

    def codificar(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparametrizar(self, mu, logvar):
        # z = μ + σ · ε, com ε ~ N(0, I). O acaso fica fora dos pesos.
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decodificar(self, z):
        h = torch.relu(self.fc3(z))
        h = torch.relu(self.fc4(h))
        return torch.sigmoid(self.fc5(h))

    def forward(self, x, amostrar=True):
        mu, logvar = self.codificar(x)
        z = self.reparametrizar(mu, logvar) if amostrar else mu
        return self.decodificar(z), mu, logvar


def perda(x, x_recon, mu, logvar):
    """||x - x'||² + KL(N(μ, σ), N(0, I)), média por imagem do lote."""
    reconstrucao = torch.sum((x - x_recon) ** 2, dim=1).mean()
    kl = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1.0 - logvar, dim=1).mean()
    return reconstrucao + kl, reconstrucao, kl


def _avaliar(modelo, x):
    modelo.eval()
    total = recon = kl = 0.0
    n = 0
    with torch.no_grad():
        for inicio in range(0, len(x), 512):
            xb = x[inicio : inicio + 512]
            # Na validação usa-se μ, sem sortear ε, para a loss não oscilar.
            xh, mu, logvar = modelo(xb, amostrar=False)
            lote_total, lote_recon, lote_kl = perda(xb, xh, mu, logvar)
            tamanho = xb.shape[0]
            total += lote_total.item() * tamanho
            recon += lote_recon.item() * tamanho
            kl += lote_kl.item() * tamanho
            n += tamanho
    return total / n, recon / n, kl / n


def treinar(modelo, x_treino, x_val, epocas, paciencia, semente, epoca_inicial=1):
    """Treina com Adam. Devolve o histórico e restaura a época de menor loss de validação.

    Os pesos de entrada já contam como candidato. Se nenhuma época melhorar a
    validação, o modelo volta ao ponto de partida.
    """
    definir_semente(semente)
    otimizador = torch.optim.Adam(modelo.parameters(), lr=TAXA)
    gerador = torch.Generator()
    gerador.manual_seed(semente)
    x_treino_t = torch.from_numpy(np.ascontiguousarray(x_treino))
    x_val_t = torch.from_numpy(np.ascontiguousarray(x_val))
    n = x_treino_t.shape[0]
    historico = []
    melhor, _, _ = _avaliar(modelo, x_val_t)
    melhor_estado = {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()}
    espera = 0

    for epoca in range(epoca_inicial, epoca_inicial + epocas):
        modelo.train()
        ordem = torch.randperm(n, generator=gerador)
        soma_total = soma_recon = soma_kl = 0.0
        vistos = 0
        for inicio in range(0, n, BATCH):
            idx = ordem[inicio : inicio + BATCH]
            xb = x_treino_t[idx]
            otimizador.zero_grad()
            xh, mu, logvar = modelo(xb, amostrar=True)
            lote_total, lote_recon, lote_kl = perda(xb, xh, mu, logvar)
            lote_total.backward()
            otimizador.step()
            tamanho = xb.shape[0]
            soma_total += lote_total.item() * tamanho
            soma_recon += lote_recon.item() * tamanho
            soma_kl += lote_kl.item() * tamanho
            vistos += tamanho

        val_total, val_recon, val_kl = _avaliar(modelo, x_val_t)
        registro = {
            "epoca": epoca,
            "treino_total": soma_total / vistos,
            "treino_recon": soma_recon / vistos,
            "treino_kl": soma_kl / vistos,
            "val_total": val_total,
            "val_recon": val_recon,
            "val_kl": val_kl,
        }
        historico.append(registro)
        print(
            f"  época {epoca:02d}  treino {registro['treino_total']:.3f}  "
            f"val {val_total:.3f}  (recon {val_recon:.3f} + kl {val_kl:.3f})",
            flush=True,
        )

        if val_total < melhor:
            melhor = val_total
            melhor_estado = {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()}
            espera = 0
        else:
            espera += 1
            if paciencia is not None and espera >= paciencia:
                print(f"  parada antecipada na época {epoca}", flush=True)
                break

    modelo.load_state_dict(melhor_estado)
    return historico, melhor


def codificar(modelo, x):
    """Devolve μ de cada imagem. O gráfico usa o centro da nuvem, não um sorteio."""
    modelo.eval()
    medias = []
    xt = torch.from_numpy(np.ascontiguousarray(x))
    with torch.no_grad():
        for inicio in range(0, len(xt), 512):
            mu, _ = modelo.codificar(xt[inicio : inicio + 512])
            medias.append(mu.numpy())
    return np.concatenate(medias, axis=0)


def pca_2d(z):
    """Projeção nas duas direções de maior espalhamento e a fração da variância em cada uma."""
    centrado = z - z.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(centrado, full_matrices=False)
    variancia = (s ** 2) / (centrado.shape[0] - 1)
    razao = variancia / variancia.sum()
    projecao = centrado @ vt[:2].T
    return projecao, razao


def _nome_largura(largura):
    return f"{largura[0]}-{largura[1]}"


def buscar(dados):
    """Dois treinos curtos por dimensão latente. O critério é a menor loss total na validação."""
    ranking = []
    estados = {}
    historias = {}
    for dim in DIMENSOES:
        for largura in LARGURAS:
            print(f"\nBusca  z={dim}  largura {_nome_largura(largura)}", flush=True)
            definir_semente(SEMENTE)
            modelo = VAE(largura, dim)
            historico, melhor = treinar(
                modelo, dados["x_treino"], dados["x_val"], EPOCAS_BUSCA, paciencia=None, semente=SEMENTE
            )
            vencedora = min(historico, key=lambda h: h["val_total"])
            ranking.append(
                {
                    "z": dim,
                    "largura": list(largura),
                    "val_total": vencedora["val_total"],
                    "val_recon": vencedora["val_recon"],
                    "val_kl": vencedora["val_kl"],
                    "epoca_melhor": vencedora["epoca"],
                }
            )
            estados[(dim, largura)] = {k: v.clone() for k, v in modelo.state_dict().items()}
            historias[(dim, largura)] = historico
    return ranking, estados, historias


def escolher(ranking):
    escolhidos = {}
    for dim in DIMENSOES:
        candidatos = [item for item in ranking if item["z"] == dim]
        escolhidos[dim] = min(candidatos, key=lambda item: item["val_total"])
    return escolhidos


def continuar_treino(modelo, dados, epocas, paciencia, semente, epoca_inicial):
    return treinar(
        modelo,
        dados["x_treino"],
        dados["x_val"],
        epocas,
        paciencia,
        semente,
        epoca_inicial=epoca_inicial,
    )


def salvar_historico(caminho, historico):
    campos = list(historico[0].keys())
    with caminho.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(historico)


def plotar_curvas(historico, titulo, caminho):
    epocas = [h["epoca"] for h in historico]
    fig, eixos = plt.subplots(1, 3, figsize=(12, 3.6), sharex=True)
    series = (
        ("total", "Loss total"),
        ("recon", "Reconstrução"),
        ("kl", "KL"),
    )
    for eixo, (chave, nome) in zip(eixos, series):
        eixo.plot(epocas, [h[f"treino_{chave}"] for h in historico], label="treino")
        eixo.plot(epocas, [h[f"val_{chave}"] for h in historico], label="validação")
        eixo.set_title(nome)
        eixo.set_xlabel("época")
        eixo.legend()
        eixo.grid(True, alpha=0.3)
    fig.suptitle(titulo)
    fig.tight_layout()
    fig.savefig(caminho, dpi=140)
    plt.close(fig)


def plotar_dispersao(pontos, rotulos, titulo, caminho, xlabel, ylabel, variancia=None):
    fig, eixo = plt.subplots(figsize=(8.2, 6.4))
    mapa = plt.get_cmap("tab10")
    for classe in range(10):
        mascara = rotulos == classe
        eixo.scatter(
            pontos[mascara, 0],
            pontos[mascara, 1],
            s=8,
            alpha=0.55,
            color=mapa(classe),
            label=NOMES[classe],
            linewidths=0,
        )
    eixo.set_title(titulo)
    eixo.set_xlabel(xlabel)
    eixo.set_ylabel(ylabel)
    eixo.legend(
        markerscale=2,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=8,
        borderaxespad=0,
    )
    eixo.grid(True, alpha=0.25)
    if variancia is not None:
        texto = (
            f"PC1 {variancia[0] * 100:.1f}%\n"
            f"PC2 {variancia[1] * 100:.1f}%\n"
            f"soma {variancia[:2].sum() * 100:.1f}%"
        )
        eixo.text(
            0.02,
            0.98,
            texto,
            transform=eixo.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )
    fig.tight_layout()
    fig.savefig(caminho, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plotar_reconstrucao(modelo, x, caminho):
    modelo.eval()
    xt = torch.from_numpy(np.ascontiguousarray(x[:10]))
    with torch.no_grad():
        xh, _, _ = modelo(xt, amostrar=False)
    original = xt.numpy().reshape(-1, 28, 28)
    reconstruida = xh.numpy().reshape(-1, 28, 28)
    fig, eixos = plt.subplots(2, 10, figsize=(10, 2.4))
    for i in range(10):
        eixos[0, i].imshow(original[i], cmap="gray", vmin=0, vmax=1)
        eixos[1, i].imshow(reconstruida[i], cmap="gray", vmin=0, vmax=1)
        eixos[0, i].axis("off")
        eixos[1, i].axis("off")
    eixos[0, 0].set_ylabel("original", fontsize=8)
    eixos[1, 0].set_ylabel("VAE", fontsize=8)
    fig.tight_layout()
    fig.savefig(caminho, dpi=140)
    plt.close(fig)


def plotar_grade_latente(modelo, caminho, passos=12):
    """Caminha no plano z de -3 a 3 e decodifica cada ponto. Só faz sentido com z = 2."""
    eixo = np.linspace(-3, 3, passos)
    grade = np.zeros((passos * 28, passos * 28), dtype=np.float32)
    modelo.eval()
    with torch.no_grad():
        for i, z2 in enumerate(eixo):
            for j, z1 in enumerate(eixo):
                z = torch.tensor([[z1, z2]], dtype=torch.float32)
                pixel = modelo.decodificar(z).numpy().reshape(28, 28)
                grade[i * 28 : (i + 1) * 28, j * 28 : (j + 1) * 28] = pixel
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.imshow(grade, cmap="gray", vmin=0, vmax=1)
    ax.set_title("Plano latente 2D, de −3 a 3")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(caminho, dpi=140)
    plt.close(fig)


def plotar_amostras(modelo, caminho, n=8):
    """Sorteia z ~ N(0, I) e decodifica. É a geração pedida pelo espaço de 8 dimensões."""
    modelo.eval()
    with torch.no_grad():
        z = torch.randn(n * n, modelo.dim_latente)
        pixels = modelo.decodificar(z).numpy().reshape(n * n, 28, 28)
    fig, eixos = plt.subplots(n, n, figsize=(6.2, 6.2))
    for i, eixo in enumerate(eixos.ravel()):
        eixo.imshow(pixels[i], cmap="gray", vmin=0, vmax=1)
        eixo.axis("off")
    fig.suptitle(f"Amostras de N(0, I) com z = {modelo.dim_latente}")
    fig.tight_layout()
    fig.savefig(caminho, dpi=140)
    plt.close(fig)


def main():
    RESULTADOS.mkdir(exist_ok=True)
    print("Carregando CSV...", flush=True)
    dados = carregar_dados()
    print(
        f"treino {len(dados['x_treino'])}  val {len(dados['x_val'])}  teste {len(dados['x_teste'])}",
        flush=True,
    )

    ranking, estados, historias = buscar(dados)
    escolhidos = escolher(ranking)
    with (RESULTADOS / "busca.csv").open("w", newline="", encoding="utf-8") as arquivo:
        campos = ["z", "largura", "val_total", "val_recon", "val_kl", "epoca_melhor"]
        escritor = csv.DictWriter(arquivo, fieldnames=campos)
        escritor.writeheader()
        for item in ranking:
            escritor.writerow(
                {
                    "z": item["z"],
                    "largura": "-".join(str(v) for v in item["largura"]),
                    "val_total": f"{item['val_total']:.6f}",
                    "val_recon": f"{item['val_recon']:.6f}",
                    "val_kl": f"{item['val_kl']:.6f}",
                    "epoca_melhor": item["epoca_melhor"],
                }
            )

    resumo = {"busca": ranking, "modelos": {}}
    for dim in DIMENSOES:
        escolha = escolhidos[dim]
        largura = tuple(escolha["largura"])
        print(
            f"\nVencedor z={dim}: {_nome_largura(largura)}  "
            f"val {escolha['val_total']:.3f}. Continuando o treino.",
            flush=True,
        )
        modelo = VAE(largura, dim)
        modelo.load_state_dict(estados[(dim, largura)])
        historico_busca = historias[(dim, largura)]
        print(
            f"Continuação a partir da época {escolha['epoca_melhor']}, "
            f"com parada antecipada.",
            flush=True,
        )
        historico_extra, melhor = continuar_treino(
            modelo,
            dados,
            EPOCAS_RETREINO,
            PACIENCIA,
            SEMENTE + dim,
            epoca_inicial=escolha["epoca_melhor"] + 1,
        )
        historico = historico_busca[: escolha["epoca_melhor"]] + historico_extra
        torch.save(
            {
                "largura": largura,
                "dim_latente": dim,
                "estado": modelo.state_dict(),
                "val_total_busca": escolha["val_total"],
                "val_total_final": melhor,
            },
            RESULTADOS / f"vae_z{dim}.pt",
        )
        salvar_historico(RESULTADOS / f"historico_z{dim}.csv", historico)
        plotar_curvas(
            historico,
            f"Retreino  z={dim}  largura {_nome_largura(largura)}",
            RESULTADOS / f"curvas_z{dim}.png",
        )
        plotar_reconstrucao(modelo, dados["x_teste"], RESULTADOS / f"reconstrucao_z{dim}.png")
        mu = codificar(modelo, dados["x_teste"])
        np.savez(RESULTADOS / f"latente_z{dim}.npz", mu=mu, y=dados["y_teste"])
        if dim == 2:
            plotar_dispersao(
                mu,
                dados["y_teste"],
                "Espaço latente 2D no conjunto de teste",
                RESULTADOS / "latente_2d.png",
                xlabel="z1",
                ylabel="z2",
            )
            plotar_grade_latente(modelo, RESULTADOS / "grade_latente_2d.png")
            variancia = None
        else:
            projecao, razao = pca_2d(mu)
            plotar_dispersao(
                projecao,
                dados["y_teste"],
                "PCA do espaço latente 8D no conjunto de teste",
                RESULTADOS / "latente_8d_pca.png",
                xlabel="PC1",
                ylabel="PC2",
                variancia=razao,
            )
            plotar_amostras(modelo, RESULTADOS / "amostras_z8.png")
            variancia = razao[:2].tolist()
        resumo["modelos"][str(dim)] = {
            "largura": list(largura),
            "val_total_busca": escolha["val_total"],
            "val_recon_busca": escolha["val_recon"],
            "val_kl_busca": escolha["val_kl"],
            "val_total_final": melhor,
            "epocas_retreino": len(historico),
            "variancia_pca": variancia,
        }

    def converter(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, tuple):
            return list(obj)
        raise TypeError(type(obj))

    (RESULTADOS / "resumo.json").write_text(
        json.dumps(resumo, indent=2, ensure_ascii=False, default=converter),
        encoding="utf-8",
    )
    print("\nConcluído. Arquivos em", RESULTADOS, flush=True)


if __name__ == "__main__":
    main()
