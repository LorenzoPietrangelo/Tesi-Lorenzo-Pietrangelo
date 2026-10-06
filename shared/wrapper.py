# Contratto dei wrapper dei modelli: load() restituisce un Handle, run() elabora un video.


# Modello caricato, con device, unità di misura e parametri da salvare nei file.
class Handle:
    # Memorizza il modello e i parametri da registrare negli npz.
    def __init__(self, model, device="cpu", units=None, **extra):
        self.model = model
        self.device = device
        self.units = units
        self.extra = extra

    # Rilascia le risorse native del modello (di default non fa nulla).
    def close(self):
        pass

    # Permette di usare l'handle in un blocco `with`.
    def __enter__(self):
        return self

    # Chiude il modello all'uscita dal blocco `with`.
    def __exit__(self, *_exc):
        self.close()
        return False
