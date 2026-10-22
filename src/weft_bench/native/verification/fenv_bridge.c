#include <fenv.h>
#include <stdlib.h>

/* The platform owns both fenv_t and the FE_* encodings. */
int wb_fe_mode_constant(int index) {
    const int modes[] = {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO};
    return index >= 0 && index < 4 ? modes[index] : -1;
}

int wb_fe_mode(void) { return fegetround(); }
int wb_fe_set(int mode) { return fesetround(mode); }
int wb_fe_flags(void) { return fetestexcept(FE_ALL_EXCEPT); }

void *wb_fe_save(void) {
    fenv_t *saved = malloc(sizeof(*saved));
    if (saved && fegetenv(saved) != 0) {
        free(saved);
        return NULL;
    }
    return saved;
}

int wb_fe_restore(void *saved) {
    int status = fesetenv((const fenv_t *)saved);
    free(saved);
    return status;
}
